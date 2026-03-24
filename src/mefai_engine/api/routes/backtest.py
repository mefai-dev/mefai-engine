"""Backtest endpoints - run and retrieve backtest results."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mefai_engine.app import get_state

router = APIRouter()


class BacktestRequest(BaseModel):
    """Backtest run parameters."""
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"
    initial_capital: float = 10000.0
    max_position_pct: float = 10.0
    leverage: int = 1
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0


@router.post("/backtest/run")
async def run_backtest(req: BacktestRequest) -> dict[str, Any]:
    """Run a backtest with specified parameters.

    Fetches historical data from exchange and runs through the
    feature pipeline + model ensemble + vectorized backtester.
    """
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    import numpy as np
    from datetime import datetime, timezone
    from mefai_engine.constants import ExchangeID
    from mefai_engine.features.pipeline import FeaturePipeline
    from mefai_engine.backtest.engine import VectorizedBacktest, BacktestConfig
    from mefai_engine.backtest.stats import format_report

    # Fetch historical data
    for eid in ExchangeID:
        exchange = factory.get(eid)
        if not exchange:
            continue

        try:
            from mefai_engine.data.collector import DataCollector
            collector = DataCollector(exchange)

            start_dt = datetime.fromisoformat(req.start_date).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(req.end_date).replace(tzinfo=timezone.utc)

            candles = await collector.fetch_historical(
                symbol=req.symbol.upper(),
                timeframe=req.timeframe,
                start=start_dt,
                end=end_dt,
            )

            if len(candles) < 100:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only {len(candles)} candles available. Need at least 100.",
                )

            # Compute features
            config = state.get("config")
            pipeline = FeaturePipeline(config.features.enabled if config else None)

            raw = {
                "open": np.array([c.open for c in candles]),
                "high": np.array([c.high for c in candles]),
                "low": np.array([c.low for c in candles]),
                "close": np.array([c.close for c in candles]),
                "volume": np.array([c.volume for c in candles]),
            }
            features = pipeline.compute(raw)

            # Generate signals from models (or use simple strategy for demo)
            closes = raw["close"]
            rsi = features.get("rsi_14")
            macd = features.get("macd_12_26_9")

            signals = np.zeros(len(closes))
            if rsi is not None and macd is not None:
                for i in range(1, len(closes)):
                    if not np.isnan(rsi[i]) and not np.isnan(macd[i]):
                        if rsi[i] < 30 and macd[i] > 0:
                            signals[i] = 1  # Long
                        elif rsi[i] > 70 and macd[i] < 0:
                            signals[i] = -1  # Short
                        else:
                            signals[i] = signals[i - 1]  # Hold

            # Run backtest
            bt_config = BacktestConfig(
                initial_capital=req.initial_capital,
                maker_fee_bps=req.maker_fee_bps,
                taker_fee_bps=req.taker_fee_bps,
                max_position_pct=req.max_position_pct,
                leverage=req.leverage,
            )

            engine = VectorizedBacktest(bt_config)
            result = engine.run(signals=signals, prices=closes)
            report = format_report(result)

            return {
                "symbol": req.symbol.upper(),
                "timeframe": req.timeframe,
                "period": f"{req.start_date} to {req.end_date}",
                "candles": len(candles),
                "results": {
                    "total_return_pct": result.total_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "calmar_ratio": result.calmar_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                    "expectancy": result.expectancy,
                    "recovery_factor": result.recovery_factor,
                },
                "equity_curve_length": len(result.equity_curve),
                "report": report,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=503, detail="No exchange available")

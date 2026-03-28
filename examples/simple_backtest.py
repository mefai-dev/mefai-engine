"""Simple backtest example using synthetic OHLCV data.

This script demonstrates the full MEFAI Engine backtest workflow:
1. Generate synthetic price data using numpy
2. Compute technical features via the feature pipeline
3. Generate trading signals from RSI
4. Run the vectorized backtest engine
5. Print performance results

No exchange connection or database is required. Everything runs locally
using only numpy and the MEFAI Engine core modules.
"""

from __future__ import annotations

import numpy as np

from mefai_engine.backtest.engine import BacktestConfig, VectorizedBacktest
from mefai_engine.features.pipeline import FeaturePipeline


def generate_synthetic_ohlcv(
    n_bars: int = 5000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate realistic synthetic OHLCV data.

    Produces a random walk with drift and volatility clustering
    that resembles real crypto price action.
    """
    rng = np.random.default_rng(seed)

    # Random walk with slight upward drift
    log_returns = rng.normal(loc=0.0001, scale=0.02, size=n_bars)

    # Add volatility clustering via a simple GARCH-like effect
    vol = np.ones(n_bars) * 0.02
    for i in range(1, n_bars):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * abs(log_returns[i - 1])
        log_returns[i] *= vol[i] / 0.02

    close = 40000.0 * np.exp(np.cumsum(log_returns))

    # Generate OHLC from close
    noise = rng.uniform(0.001, 0.01, size=n_bars)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))

    volume = rng.lognormal(mean=15, sigma=0.5, size=n_bars)

    return {
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def rsi_signal_generator(
    rsi: np.ndarray,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate trading signals from RSI values.

    Returns signal array (1=long -1=short 0=flat) and confidence array.
    """
    n = len(rsi)
    signals = np.zeros(n)
    confidences = np.zeros(n)

    for i in range(1, n):
        if np.isnan(rsi[i]):
            continue

        if rsi[i] < oversold:
            signals[i] = 1.0  # Long on oversold
            confidences[i] = min((oversold - rsi[i]) / oversold + 0.6, 1.0)
        elif rsi[i] > overbought:
            signals[i] = -1.0  # Short on overbought
            confidences[i] = min((rsi[i] - overbought) / (100 - overbought) + 0.6, 1.0)
        else:
            # Hold previous direction if between thresholds
            signals[i] = signals[i - 1]
            confidences[i] = confidences[i - 1] * 0.95

    return signals, confidences


def main() -> None:
    """Run the complete backtest pipeline."""
    print("=" * 60)
    print("  MEFAI Engine - Simple Backtest Example")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n[1] Generating synthetic OHLCV data (5000 bars)...")
    raw = generate_synthetic_ohlcv(n_bars=5000, seed=42)
    print(f"    Price range: {raw['close'].min():.2f} to {raw['close'].max():.2f}")

    # Step 2: Compute features
    print("\n[2] Computing technical features...")
    pipeline = FeaturePipeline(enabled_features=[
        "rsi_14",
        "sma_20",
        "sma_50",
        "ema_10",
        "ema_50",
        "atr_14",
        "bollinger_width_20",
        "obv",
    ])
    features = pipeline.compute(raw)
    print(f"    Computed {len(features)} features")

    # Step 3: Generate signals from RSI
    print("\n[3] Generating RSI-based trading signals...")
    rsi = features["rsi_14"]
    signals, confidences = rsi_signal_generator(rsi)
    long_count = int(np.sum(signals == 1))
    short_count = int(np.sum(signals == -1))
    flat_count = int(np.sum(signals == 0))
    print(f"    Long bars: {long_count}  Short bars: {short_count}  Flat bars: {flat_count}")

    # Step 4: Run backtest
    print("\n[4] Running vectorized backtest...")
    config = BacktestConfig(
        initial_capital=10000.0,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        slippage_bps=3.0,
        max_position_pct=10.0,
        leverage=2,
    )
    bt = VectorizedBacktest(config)
    result = bt.run(signals, raw["close"], confidences)

    # Step 5: Print results
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:       {result.total_return_pct:>10.2f}%")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:>10.3f}")
    print(f"  Sortino Ratio:      {result.sortino_ratio:>10.3f}")
    print(f"  Max Drawdown:       {result.max_drawdown_pct:>10.2f}%")
    print(f"  Win Rate:           {result.win_rate:>10.1%}")
    print(f"  Profit Factor:      {result.profit_factor:>10.3f}")
    print(f"  Total Trades:       {result.total_trades:>10d}")
    print(f"  Calmar Ratio:       {result.calmar_ratio:>10.3f}")
    print(f"  Recovery Factor:    {result.recovery_factor:>10.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

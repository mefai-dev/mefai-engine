"""Data endpoints - historical candles and features and news."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from mefai_engine.api.middleware import check_rate_limit, require_api_key
from mefai_engine.app import get_state

router = APIRouter(dependencies=[Depends(require_api_key), Depends(check_rate_limit)])


@router.get("/candles/{symbol}")
async def get_candles(
    symbol: str,
    timeframe: str = Query(default="1h", description="Candle timeframe"),
    limit: int = Query(default=200, ge=1, le=1500, description="Number of candles"),
) -> dict[str, Any]:
    """Get historical candles from exchange."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    from mefai_engine.constants import ExchangeID
    for eid in ExchangeID:
        exchange = factory.get(eid)
        if exchange:
            try:
                candles = await exchange.get_ohlcv(symbol.upper(), timeframe, limit=limit)
                return {
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "count": len(candles),
                    "candles": [
                        {
                            "t": c.timestamp.isoformat(),
                            "o": c.open,
                            "h": c.high,
                            "l": c.low,
                            "c": c.close,
                            "v": c.volume,
                        }
                        for c in candles
                    ],
                }
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))

    raise HTTPException(status_code=503, detail="No exchange available")


@router.get("/features/{symbol}")
async def get_features(
    symbol: str,
    timeframe: str = Query(default="1h"),
) -> dict[str, Any]:
    """Get computed features for a symbol (from cache or compute live)."""
    state = get_state()
    cache = state.get("cache")

    # Try cache first
    if cache:
        cached = await cache.get_features(symbol.upper(), timeframe)
        if cached:
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "source": "cache",
                "features": cached,
            }

    # Compute from live data
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    from mefai_engine.constants import ExchangeID
    import numpy as np

    for eid in ExchangeID:
        exchange = factory.get(eid)
        if exchange:
            try:
                candles = await exchange.get_ohlcv(symbol.upper(), timeframe, limit=300)
                if len(candles) < 50:
                    raise HTTPException(status_code=400, detail="Not enough data for features")

                opens = np.array([c.open for c in candles])
                highs = np.array([c.high for c in candles])
                lows = np.array([c.low for c in candles])
                closes = np.array([c.close for c in candles])
                volumes = np.array([c.volume for c in candles])

                config = state.get("config")
                enabled = config.features.enabled if config else None

                from mefai_engine.features.pipeline import FeaturePipeline
                pipeline = FeaturePipeline(enabled)

                raw = {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                }

                computed = pipeline.compute(raw)
                # Return last value of each feature
                result = {}
                for name, arr in computed.items():
                    val = float(arr[-1]) if not np.isnan(arr[-1]) else None
                    if val is not None:
                        result[name] = round(val, 6)

                # Cache the result
                if cache:
                    await cache.set_features(symbol.upper(), timeframe, result)

                return {
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "source": "computed",
                    "feature_count": len(result),
                    "features": result,
                }

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))

    raise HTTPException(status_code=503, detail="No exchange available")


@router.get("/news")
async def get_news(
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Get latest crypto news from aggregated sources."""
    from mefai_engine.data.news_fetcher import NewsAggregator

    aggregator = NewsAggregator()
    news = await aggregator.fetch_all(limit=limit)

    return {
        "count": len(news),
        "news": [
            {
                "title": item.title,
                "source": item.source,
                "url": item.url,
                "published_at": item.published_at.isoformat(),
                "currencies": item.currencies,
                "sentiment_hint": item.sentiment_hint,
            }
            for item in news
        ],
    }


@router.get("/sentiment/{symbol}")
async def get_sentiment(symbol: str) -> dict[str, Any]:
    """Get current sentiment score for a symbol."""
    state = get_state()
    cache = state.get("cache")

    # Check cache
    if cache:
        score = await cache.get_sentiment(symbol.upper())
        if score is not None:
            return {
                "symbol": symbol.upper(),
                "score": round(score, 4),
                "label": "bullish" if score > 0.1 else ("bearish" if score < -0.1 else "neutral"),
                "source": "cache",
            }

    # Compute fresh from news
    from mefai_engine.data.news_fetcher import CryptoPanicFetcher
    fetcher = CryptoPanicFetcher()
    news_items = await fetcher.fetch(currencies=symbol[:3].upper(), limit=10)

    if not news_items:
        return {
            "symbol": symbol.upper(),
            "score": 0.0,
            "label": "neutral",
            "source": "no_data",
        }

    try:
        from mefai_engine.models.nlp.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        analyzer.load()
        titles = [item.title for item in news_items]
        scores = analyzer.score_batch(titles)
        import numpy as np
        avg = float(np.mean(scores))

        if cache:
            await cache.set_sentiment(symbol.upper(), avg)

        return {
            "symbol": symbol.upper(),
            "score": round(avg, 4),
            "label": "bullish" if avg > 0.1 else ("bearish" if avg < -0.1 else "neutral"),
            "headlines_analyzed": len(titles),
            "source": "finbert",
        }
    except ImportError:
        return {
            "symbol": symbol.upper(),
            "score": 0.0,
            "label": "neutral",
            "source": "model_unavailable",
            "note": "Install transformers package for sentiment analysis",
        }

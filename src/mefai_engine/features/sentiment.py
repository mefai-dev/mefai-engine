"""Sentiment features from news and social data.

Aggregates sentiment scores into trading-ready features.
"""

from __future__ import annotations

import numpy as np

from mefai_engine.features.registry import feature


@feature(name="news_sentiment", depends_on=["sentiment_scores"], category="sentiment")
def news_sentiment(sentiment_scores: np.ndarray) -> np.ndarray:
    """Raw news sentiment score (-1 to +1)."""
    return sentiment_scores


@feature(name="sentiment_momentum", depends_on=["sentiment_scores"], params={"period": 12}, category="sentiment")
def sentiment_momentum(sentiment_scores: np.ndarray, period: int = 12) -> np.ndarray:
    """Change in sentiment over N periods (momentum)."""
    out = np.full_like(sentiment_scores, np.nan)
    out[period:] = sentiment_scores[period:] - sentiment_scores[:-period]
    return out


@feature(name="sentiment_volatility", depends_on=["sentiment_scores"], params={"period": 24}, category="sentiment")
def sentiment_volatility(sentiment_scores: np.ndarray, period: int = 24) -> np.ndarray:
    """Rolling standard deviation of sentiment (disagreement/uncertainty)."""
    n = len(sentiment_scores)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.std(sentiment_scores[i - period + 1: i + 1])
    return out


@feature(name="sentiment_price_divergence", depends_on=["sentiment_scores", "close"], params={"period": 24}, category="sentiment")
def sentiment_price_divergence(sentiment_scores: np.ndarray, close: np.ndarray, period: int = 24) -> np.ndarray:
    """Divergence between sentiment and price direction.

    Positive = sentiment bullish but price falling (potential reversal up).
    Negative = sentiment bearish but price rising (potential reversal down).
    """
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n):
        price_change = (close[i] - close[i - period]) / close[i - period]
        avg_sentiment = np.mean(sentiment_scores[i - period + 1: i + 1])
        out[i] = avg_sentiment - np.sign(price_change) * min(abs(price_change) * 10, 1.0)
    return out

"""FinBERT-based financial sentiment analysis.

Scores financial text on a scale of -1.0 (bearish) to +1.0 (bullish).
Supports batch processing with caching.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from mefai_engine.constants import Direction
from mefai_engine.models.base import BasePredictor
from mefai_engine.types import Prediction

logger = structlog.get_logger()


class SentimentAnalyzer:
    """FinBERT-based sentiment scorer for financial text.

    Uses ProsusAI/finbert or compatible model for sentiment classification.
    Outputs: positive/negative/neutral probabilities + overall score.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self._model_name = model_name
        self._pipeline: Any = None
        self._loaded = False

    def load(self) -> None:
        """Load the sentiment model (lazy loading)."""
        if self._loaded:
            return

        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model_name,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            self._loaded = True
            logger.info("sentiment.model_loaded", model=self._model_name)
        except Exception:
            logger.exception("sentiment.load_failed")

    def score(self, text: str) -> float:
        """Score a single text. Returns -1.0 to +1.0."""
        if not self._loaded:
            self.load()
        if self._pipeline is None:
            return 0.0

        try:
            results = self._pipeline(text[:512])
            return self._parse_score(results[0] if results else [])
        except Exception:
            logger.exception("sentiment.score_error")
            return 0.0

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score multiple texts efficiently."""
        if not self._loaded:
            self.load()
        if self._pipeline is None:
            return [0.0] * len(texts)

        try:
            truncated = [t[:512] for t in texts]
            results = self._pipeline(truncated, batch_size=8)
            return [self._parse_score(r) for r in results]
        except Exception:
            logger.exception("sentiment.batch_error")
            return [0.0] * len(texts)

    @staticmethod
    def _parse_score(result: list[dict[str, Any]]) -> float:
        """Convert FinBERT output to single score."""
        if not result:
            return 0.0

        scores = {r["label"].lower(): r["score"] for r in result}
        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        # neutral doesn't contribute to direction

        return positive - negative  # Range: -1.0 to +1.0


class SentimentPredictor(BasePredictor):
    """Wraps SentimentAnalyzer as a BasePredictor for the model registry.

    Aggregates sentiment from multiple news items into a trading signal.
    """

    model_id = "sentiment"
    model_version = "v1"

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        bullish_threshold: float = 0.3,
        bearish_threshold: float = -0.3,
        horizon_seconds: int = 14400,
    ) -> None:
        self._analyzer = SentimentAnalyzer(model_name)
        self._bullish_threshold = bullish_threshold
        self._bearish_threshold = bearish_threshold
        self._horizon_seconds = horizon_seconds
        self._trained = True  # Pre-trained model, always "trained"

    @property
    def is_trained(self) -> bool:
        return True

    def predict_from_texts(self, texts: list[str]) -> Prediction:
        """Generate prediction from a list of news headlines/texts."""
        if not texts:
            return Prediction(
                direction=Direction.FLAT,
                confidence=0.5,
                magnitude=0.0,
                horizon_seconds=self._horizon_seconds,
                model_id=self.model_id,
                model_version=self.model_version,
            )

        scores = self._analyzer.score_batch(texts)
        avg_score = float(np.mean(scores))
        abs_score = abs(avg_score)

        if avg_score > self._bullish_threshold:
            direction = Direction.LONG
        elif avg_score < self._bearish_threshold:
            direction = Direction.SHORT
        else:
            direction = Direction.FLAT

        # Confidence = how far from neutral
        confidence = min(abs_score / 0.5, 1.0) * 0.5 + 0.5  # Scale to 0.5-1.0

        return Prediction(
            direction=direction,
            confidence=confidence,
            magnitude=avg_score,
            horizon_seconds=self._horizon_seconds,
            model_id=self.model_id,
            model_version=self.model_version,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def predict(self, features: np.ndarray) -> Prediction:
        """Predict from pre-computed sentiment features."""
        if features.ndim == 1:
            avg_sentiment = float(features[0]) if len(features) > 0 else 0.0
        else:
            avg_sentiment = float(np.mean(features[:, 0])) if features.shape[1] > 0 else 0.0

        if avg_sentiment > self._bullish_threshold:
            direction = Direction.LONG
        elif avg_sentiment < self._bearish_threshold:
            direction = Direction.SHORT
        else:
            direction = Direction.FLAT

        confidence = min(abs(avg_sentiment) * 2, 1.0)

        return Prediction(
            direction=direction,
            confidence=confidence,
            magnitude=avg_sentiment,
            horizon_seconds=self._horizon_seconds,
            model_id=self.model_id,
            model_version=self.model_version,
        )

    def predict_batch(self, features: np.ndarray) -> list[Prediction]:
        return [self.predict(features[i]) for i in range(len(features))]

    def fit(self, features: np.ndarray, targets: np.ndarray, validation_split: float = 0.2) -> dict[str, float]:
        """FinBERT is pre-trained, no fitting needed."""
        return {"status": "pre-trained model, no fit required"}

    def save(self, path: Path) -> None:
        """FinBERT model is loaded from HuggingFace, no local save needed."""
        pass

    def load(self, path: Path) -> None:
        """Load triggers the HuggingFace model download."""
        self._analyzer.load()

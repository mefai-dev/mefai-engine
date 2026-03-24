"""Abstract base for all prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from mefai_engine.types import Prediction


class BasePredictor(ABC):
    """Protocol that all MEFAI prediction models must implement."""

    model_id: str
    model_version: str

    @abstractmethod
    def predict(self, features: np.ndarray) -> Prediction:
        """Generate a single prediction from feature vector."""
        ...

    @abstractmethod
    def predict_batch(self, features: np.ndarray) -> list[Prediction]:
        """Generate predictions from feature matrix (rows = samples)."""
        ...

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_split: float = 0.2,
    ) -> dict[str, float]:
        """Train the model. Returns metrics dict."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        ...

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return False

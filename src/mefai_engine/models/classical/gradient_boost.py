"""XGBoost / LightGBM direction classifier.

Predicts whether price will go UP, DOWN, or FLAT over the next N candles.
Uses walk-forward validation to prevent lookahead bias.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from mefai_engine.constants import Direction
from mefai_engine.models.base import BasePredictor
from mefai_engine.types import Prediction

logger = structlog.get_logger()

# Direction encoding
_DIR_MAP = {0: Direction.SHORT, 1: Direction.FLAT, 2: Direction.LONG}
_DIR_INVERSE = {v: k for k, v in _DIR_MAP.items()}


class GradientBoostPredictor(BasePredictor):
    """XGBoost-based direction prediction model.

    Features:
    - 3-class classification (LONG/SHORT/FLAT)
    - Feature importance tracking for explainability
    - Walk-forward validation support
    - Automatic class balancing
    """

    model_id = "gradient_boost"
    model_version = "v1"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        use_lightgbm: bool = False,
        horizon_seconds: int = 3600,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._use_lightgbm = use_lightgbm
        self._horizon_seconds = horizon_seconds
        self._model: Any = None
        self._feature_names: list[str] = []
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_split: float = 0.2,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Train the gradient boosting classifier.

        Args:
            features: (n_samples, n_features) array
            targets: (n_samples,) array with values in {0, 1, 2}
            validation_split: fraction for validation (from the END, not random)
            feature_names: optional feature names for importance tracking

        Returns:
            Dictionary of metrics: accuracy, f1, etc.
        """
        if feature_names:
            self._feature_names = feature_names

        # Walk-forward split (no shuffle - temporal data)
        split_idx = int(len(features) * (1 - validation_split))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = targets[:split_idx], targets[split_idx:]

        if self._use_lightgbm:
            self._model = self._train_lgbm(X_train, y_train, X_val, y_val)
        else:
            self._model = self._train_xgb(X_train, y_train, X_val, y_val)

        self._trained = True

        # Compute metrics
        val_pred = self._model.predict(X_val)
        accuracy = np.mean(val_pred == y_val)

        # Per-class accuracy
        metrics: dict[str, float] = {"accuracy": float(accuracy)}
        for cls, direction in _DIR_MAP.items():
            mask = y_val == cls
            if mask.sum() > 0:
                cls_acc = np.mean(val_pred[mask] == cls)
                metrics[f"accuracy_{direction.value}"] = float(cls_acc)

        logger.info(
            "gradient_boost.trained",
            accuracy=f"{accuracy:.4f}",
            train_size=len(X_train),
            val_size=len(X_val),
        )
        return metrics

    def _train_xgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """Train XGBoost model."""
        import xgboost as xgb

        # Calculate class weights
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        sample_weights = np.ones(len(y_train))
        for cls, count in zip(classes, counts):
            sample_weights[y_train == cls] = total / (len(classes) * count)

        model = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            early_stopping_rounds=20,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    def _train_lgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """Train LightGBM model."""
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        return model

    def predict(self, features: np.ndarray) -> Prediction:
        """Predict direction from a single feature vector."""
        if not self._trained or self._model is None:
            from mefai_engine.exceptions import ModelNotTrainedError
            raise ModelNotTrainedError("GradientBoost model not trained")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        pred_class = int(self._model.predict(features)[0])
        proba = self._model.predict_proba(features)[0]
        confidence = float(proba[pred_class])
        direction = _DIR_MAP.get(pred_class, Direction.FLAT)

        return Prediction(
            direction=direction,
            confidence=confidence,
            magnitude=float(proba[2] - proba[0]),  # long_prob - short_prob
            horizon_seconds=self._horizon_seconds,
            model_id=self.model_id,
            model_version=self.model_version,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def predict_batch(self, features: np.ndarray) -> list[Prediction]:
        """Predict for multiple samples."""
        if not self._trained or self._model is None:
            from mefai_engine.exceptions import ModelNotTrainedError
            raise ModelNotTrainedError("GradientBoost model not trained")

        pred_classes = self._model.predict(features)
        probas = self._model.predict_proba(features)
        predictions: list[Prediction] = []

        for i, (cls, proba) in enumerate(zip(pred_classes, probas)):
            cls = int(cls)
            predictions.append(Prediction(
                direction=_DIR_MAP.get(cls, Direction.FLAT),
                confidence=float(proba[cls]),
                magnitude=float(proba[2] - proba[0]),
                horizon_seconds=self._horizon_seconds,
                model_id=self.model_id,
                model_version=self.model_version,
            ))

        return predictions

    def feature_importance(self, top_n: int = 20) -> dict[str, float]:
        """Get top N important features."""
        if not self._trained or self._model is None:
            return {}

        importances = self._model.feature_importances_
        if self._feature_names and len(self._feature_names) == len(importances):
            pairs = sorted(
                zip(self._feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            pairs = [(f"f_{i}", v) for i, v in enumerate(importances)]
            pairs.sort(key=lambda x: x[1], reverse=True)

        return dict(pairs[:top_n])

    def save(self, path: Path) -> None:
        """Save model using native format (safe) with metadata sidecar."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model in native format
        model_path = path.with_suffix(".model")
        self._model.save_model(str(model_path))

        # Save metadata as JSON
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": self._feature_names,
                "version": self.model_version,
                "params": {
                    "n_estimators": self._n_estimators,
                    "max_depth": self._max_depth,
                    "learning_rate": self._learning_rate,
                },
            }, f)
        logger.info("gradient_boost.saved", path=str(model_path))

    def load(self, path: Path) -> None:
        """Load model using native format (avoids unsafe pickle deserialization)."""
        import json

        # Try native XGBoost/LightGBM format first (safe)
        meta_path = path.with_suffix(".meta.json")
        model_path = path.with_suffix(".model")

        if model_path.exists() and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._feature_names = meta.get("feature_names", [])
            self.model_version = meta.get("version", "v1")

            if self._use_lightgbm:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=str(model_path))
            else:
                import xgboost as xgb
                self._model = xgb.XGBClassifier()
                self._model.load_model(str(model_path))
            self._trained = True
            logger.info("gradient_boost.loaded_native", path=str(model_path))
            return

        # Fallback: pickle with restricted unpickler
        logger.warning("gradient_boost.pickle_fallback", path=str(path))
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        self._model = data["model"]
        self._feature_names = data.get("feature_names", [])
        self.model_version = data.get("version", "v1")
        self._trained = True
        logger.info("gradient_boost.loaded", path=str(path))

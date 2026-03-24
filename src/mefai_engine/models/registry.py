"""Model registry - manages all prediction models."""

from __future__ import annotations

from pathlib import Path

import structlog

from mefai_engine.models.base import BasePredictor

logger = structlog.get_logger()


class ModelRegistry:
    """Central registry for all prediction models.

    Handles model lifecycle: registration, training status,
    loading/saving, and version tracking.
    """

    def __init__(self, models_dir: str | Path = "models") -> None:
        self._models: dict[str, BasePredictor] = {}
        self._models_dir = Path(models_dir)

    def register(self, model: BasePredictor) -> None:
        """Register a model instance."""
        self._models[model.model_id] = model
        logger.info(
            "registry.registered",
            model_id=model.model_id,
            version=model.model_version,
            trained=model.is_trained,
        )

    def get(self, model_id: str) -> BasePredictor | None:
        """Get a registered model by ID."""
        return self._models.get(model_id)

    def get_all(self) -> dict[str, BasePredictor]:
        """Get all registered models."""
        return dict(self._models)

    def get_trained(self) -> dict[str, BasePredictor]:
        """Get only trained models."""
        return {k: v for k, v in self._models.items() if v.is_trained}

    def save_all(self) -> None:
        """Save all trained models to disk."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        for model_id, model in self._models.items():
            if model.is_trained:
                path = self._models_dir / f"{model_id}.bin"
                try:
                    model.save(path)
                except Exception:
                    logger.exception("registry.save_failed", model_id=model_id)

    def load_all(self) -> int:
        """Load all available models from disk. Returns count loaded."""
        loaded = 0
        for model_id, model in self._models.items():
            # Try multiple extensions
            for ext in [".bin", ".pt", ".zip", ".pkl"]:
                path = self._models_dir / f"{model_id}{ext}"
                if path.exists():
                    try:
                        model.load(path)
                        loaded += 1
                        logger.info("registry.loaded", model_id=model_id, path=str(path))
                    except Exception:
                        logger.exception("registry.load_failed", model_id=model_id)
                    break

        return loaded

    def status(self) -> dict[str, dict[str, str | bool]]:
        """Get status of all models."""
        return {
            model_id: {
                "version": model.model_version,
                "trained": model.is_trained,
            }
            for model_id, model in self._models.items()
        }

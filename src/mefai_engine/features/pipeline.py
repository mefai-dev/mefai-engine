"""Feature computation pipeline with DAG-based execution."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from mefai_engine.features.registry import get_feature, list_features, resolve_dependencies

logger = structlog.get_logger()

# Ensure all feature modules are imported so decorators register
import mefai_engine.features.microstructure  # noqa: F401
import mefai_engine.features.onchain  # noqa: F401
import mefai_engine.features.regime  # noqa: F401
import mefai_engine.features.technical  # noqa: F401


class FeaturePipeline:
    """Computes features from raw OHLCV data using registered feature functions.

    Resolves dependencies and computes in topological order.
    """

    def __init__(self, enabled_features: list[str] | None = None) -> None:
        if enabled_features is None:
            self._requested = [f.name for f in list_features()]
        else:
            self._requested = enabled_features
        self._compute_order = resolve_dependencies(self._requested)

    def compute(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute all enabled features from raw data.

        Args:
            raw: Dictionary with at least 'open', 'high', 'low', 'close', 'volume'.
                 May also include 'funding_rate_raw', 'open_interest',
                 'bid_volume', 'ask_volume', etc.

        Returns:
            Dictionary mapping feature name to computed array.
        """
        results: dict[str, np.ndarray] = dict(raw)

        for name in self._compute_order:
            if name in results:
                continue

            try:
                spec = get_feature(name)
            except KeyError:
                continue

            # Gather arguments
            args: dict[str, Any] = {}
            missing = False
            for dep in spec.depends_on:
                if dep in results:
                    args[dep] = results[dep]
                else:
                    missing = True
                    break

            if missing:
                logger.debug("feature.skipped_missing_dep", feature=name)
                continue

            args.update(spec.params)

            try:
                results[name] = spec.func(**args)
            except Exception:
                logger.exception("feature.compute_error", feature=name)

        # Return only requested features (not raw data)
        return {k: v for k, v in results.items() if k in self._requested}

    @property
    def feature_names(self) -> list[str]:
        """List of feature names that will be computed."""
        return list(self._requested)

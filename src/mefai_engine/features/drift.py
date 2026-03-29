"""Feature drift detection using PSI and KS tests.

Monitors each feature's distribution against a training baseline
and triggers alerts or model retraining when drift exceeds thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.25
    ks_alpha: float = 0.05
    n_bins: int = 10
    min_samples: int = 100
    check_interval_seconds: int = 3600


@dataclass
class FeatureDriftResult:
    """Drift analysis result for a single feature."""
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    is_drifted: bool
    drift_severity: str  # "none" or "warning" or "critical"
    timestamp: datetime = dc_field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass
class DriftReport:
    """Aggregate drift report across all features."""
    feature_results: list[FeatureDriftResult]
    total_features: int
    drifted_count: int
    critical_count: int
    warning_count: int
    should_retrain: bool
    timestamp: datetime = dc_field(default_factory=lambda: datetime.now(tz=UTC))


class FeatureDriftDetector:
    """Detects distribution drift in feature values.

    Uses two complementary methods:
    1. PSI (Population Stability Index) for overall distribution shift
    2. KS test (Kolmogorov-Smirnov) for statistical significance

    When drift exceeds thresholds the detector flags a retrain recommendation.
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        self._config = config or DriftConfig()
        self._baselines: dict[str, np.ndarray] = {}
        self._feature_names: list[str] = []
        self._baseline_histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def set_baseline(
        self,
        features: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """Set the training distribution baseline.

        Args:
            features: Training feature matrix (n_samples x n_features)
            feature_names: Optional list of feature names
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_features = features.shape[1]
        self._feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

        for i in range(n_features):
            name = self._feature_names[i]
            col = features[:, i]
            self._baselines[name] = col.copy()

            # Precompute baseline histogram
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                hist, edges = np.histogram(valid, bins=self._config.n_bins)
                self._baseline_histograms[name] = (hist.astype(float), edges)

        logger.info(
            "drift.baseline_set",
            n_features=n_features,
            n_samples=features.shape[0],
        )

    def check(
        self,
        features: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DriftReport:
        """Check current features against baseline for drift.

        Args:
            features: Current feature matrix (n_samples x n_features)
            feature_names: Optional feature names (must match baseline)

        Returns:
            DriftReport with per-feature results and retrain recommendation.
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        names = feature_names or self._feature_names
        results: list[FeatureDriftResult] = []

        for i, name in enumerate(names):
            if name not in self._baselines:
                continue

            if i >= features.shape[1]:
                break

            current = features[:, i]
            baseline = self._baselines[name]

            # Skip if insufficient samples
            valid_current = current[~np.isnan(current)]
            if len(valid_current) < self._config.min_samples:
                continue

            psi = self._compute_psi(name, valid_current)
            ks_stat, ks_p = self._compute_ks(baseline, valid_current)

            # Determine severity
            if psi >= self._config.psi_critical_threshold or ks_p < self._config.ks_alpha / 10:
                severity = "critical"
                is_drifted = True
            elif psi >= self._config.psi_warning_threshold or ks_p < self._config.ks_alpha:
                severity = "warning"
                is_drifted = True
            else:
                severity = "none"
                is_drifted = False

            results.append(FeatureDriftResult(
                feature_name=name,
                psi_score=round(psi, 6),
                ks_statistic=round(ks_stat, 6),
                ks_p_value=round(ks_p, 6),
                is_drifted=is_drifted,
                drift_severity=severity,
            ))

        drifted = [r for r in results if r.is_drifted]
        critical = [r for r in results if r.drift_severity == "critical"]
        warning = [r for r in results if r.drift_severity == "warning"]

        # Retrain if more than 30% of features have drifted or any critical
        retrain_threshold = 0.3
        drift_ratio = len(drifted) / len(results) if results else 0.0
        should_retrain = len(critical) > 0 or drift_ratio > retrain_threshold

        report = DriftReport(
            feature_results=results,
            total_features=len(results),
            drifted_count=len(drifted),
            critical_count=len(critical),
            warning_count=len(warning),
            should_retrain=should_retrain,
        )

        if should_retrain:
            logger.warning(
                "drift.retrain_recommended",
                drifted=len(drifted),
                critical=len(critical),
                total=len(results),
            )
        else:
            logger.debug(
                "drift.check_ok",
                drifted=len(drifted),
                total=len(results),
            )

        return report

    def _compute_psi(self, feature_name: str, current: np.ndarray) -> float:
        """Compute Population Stability Index.

        PSI measures the shift between two distributions:
        PSI = SUM( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

        PSI < 0.1: no significant drift
        0.1 <= PSI < 0.25: moderate drift
        PSI >= 0.25: significant drift
        """
        if feature_name not in self._baseline_histograms:
            return 0.0

        baseline_hist, edges = self._baseline_histograms[feature_name]
        current_hist, _ = np.histogram(current, bins=edges)
        current_hist = current_hist.astype(float)

        # Normalize to proportions
        baseline_total = baseline_hist.sum()
        current_total = current_hist.sum()

        if baseline_total == 0 or current_total == 0:
            return 0.0

        expected = baseline_hist / baseline_total
        actual = current_hist / current_total

        # Avoid division by zero with small epsilon
        eps = 1e-8
        expected = np.clip(expected, eps, None)
        actual = np.clip(actual, eps, None)

        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        return max(psi, 0.0)

    @staticmethod
    def _compute_ks(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
        """Compute Kolmogorov-Smirnov test statistic and p-value.

        Returns:
            Tuple of (ks_statistic and p_value).
        """
        try:
            from scipy import stats
            result = stats.ks_2samp(baseline, current)
            return float(result.statistic), float(result.pvalue)
        except ImportError:
            # Fallback: manual KS computation without scipy
            return _manual_ks_2samp(baseline, current)


def _manual_ks_2samp(
    sample1: np.ndarray, sample2: np.ndarray
) -> tuple[float, float]:
    """Manual two-sample KS test when scipy is not available."""
    n1 = len(sample1)
    n2 = len(sample2)

    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    all_values = np.sort(np.concatenate([sample1, sample2]))
    cdf1 = np.searchsorted(np.sort(sample1), all_values, side="right") / n1
    cdf2 = np.searchsorted(np.sort(sample2), all_values, side="right") / n2

    ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

    # Approximate p-value using asymptotic formula
    n_eff = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (n_eff + 0.12 + 0.11 / n_eff) * ks_stat

    if lam < 0.001:
        p_value = 1.0
    else:
        # Kolmogorov distribution approximation
        p_value = 2.0 * np.exp(-2.0 * lam * lam)
        p_value = float(np.clip(p_value, 0.0, 1.0))

    return ks_stat, p_value

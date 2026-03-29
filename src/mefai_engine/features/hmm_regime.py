"""Hidden Markov Model based market regime detection.

Fits a Gaussian HMM on returns and volatility and volume features
to identify latent market states (bull / bear / ranging / high_vol / low_vol).
Supports online prediction for real time regime classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class RegimeState:
    """Current regime detection result."""
    regime_id: int
    regime_label: str
    probability: float
    all_probabilities: dict[str, float]
    timestamp: datetime = dc_field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass
class HMMConfig:
    """Configuration for HMM regime detector."""
    n_states: int = 4
    n_iterations: int = 100
    convergence_threshold: float = 1e-6
    random_seed: int = 42
    lookback_window: int = 252
    volatility_window: int = 20
    volume_window: int = 20
    regime_labels: list[str] = dc_field(default_factory=lambda: [
        "bull", "bear", "ranging", "high_volatility"
    ])


class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection.

    States are learned from returns and realized volatility and
    normalized volume. After fitting the detector can predict
    the current regime from new observations in an online fashion.
    """

    def __init__(self, config: HMMConfig | None = None) -> None:
        self._config = config or HMMConfig()
        self._model: Any = None
        self._fitted = False
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._regime_mapping: dict[int, str] = {}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _prepare_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute HMM input features from price and volume arrays.

        Features:
        1. Log returns
        2. Rolling realized volatility
        3. Normalized volume (if available)
        """
        cfg = self._config

        # Log returns
        log_prices = np.log(np.maximum(prices, 1e-10))
        returns = np.diff(log_prices)

        # Realized volatility (rolling std of returns)
        vol_window = min(cfg.volatility_window, len(returns))
        volatility = np.zeros(len(returns))
        for i in range(vol_window, len(returns)):
            volatility[i] = np.std(returns[i - vol_window:i])
        # Fill initial values with the first computed volatility
        if vol_window < len(returns):
            volatility[:vol_window] = volatility[vol_window]

        feature_list = [returns.reshape(-1, 1), volatility.reshape(-1, 1)]

        # Normalized volume
        if volumes is not None and len(volumes) > 1:
            vol_data = volumes[1:]  # Align with returns
            vol_window_size = min(cfg.volume_window, len(vol_data))
            vol_sma = np.convolve(
                vol_data, np.ones(vol_window_size) / vol_window_size, mode="same"
            )
            vol_ratio = np.where(vol_sma > 0, vol_data / vol_sma, 1.0)
            feature_list.append(vol_ratio.reshape(-1, 1))

        features = np.hstack(feature_list)

        # Remove NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)

        return features

    def fit(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit the HMM on historical price data.

        Args:
            prices: Historical close prices
            volumes: Optional volume array (same length as prices)

        Returns:
            Dictionary with fitting statistics.
        """
        features = self._prepare_features(prices, volumes)

        if len(features) < self._config.n_states * 10:
            raise ValueError(
                f"Insufficient data: need at least {self._config.n_states * 10} "
                f"samples but got {len(features)}"
            )

        # Standardize features
        self._scaler_mean = np.mean(features, axis=0)
        self._scaler_std = np.std(features, axis=0)
        self._scaler_std = np.where(self._scaler_std > 0, self._scaler_std, 1.0)
        scaled = (features - self._scaler_mean) / self._scaler_std

        try:
            from hmmlearn.hmm import GaussianHMM

            self._model = GaussianHMM(
                n_components=self._config.n_states,
                covariance_type="full",
                n_iter=self._config.n_iterations,
                tol=self._config.convergence_threshold,
                random_state=self._config.random_seed,
            )
            self._model.fit(scaled)
            self._fitted = True

        except ImportError:
            # Fallback: use a simple Gaussian mixture approach
            logger.warning("hmm.hmmlearn_not_installed_using_fallback")
            self._model = _SimpleFallbackHMM(
                n_states=self._config.n_states,
                seed=self._config.random_seed,
            )
            self._model.fit(scaled)
            self._fitted = True

        # Map states to labels based on mean return of each state
        self._map_regimes_to_labels(features)

        stats = {
            "n_states": self._config.n_states,
            "n_samples": len(features),
            "regime_mapping": self._regime_mapping,
        }

        logger.info("hmm.fitted", **stats)
        return stats

    def _map_regimes_to_labels(self, features: np.ndarray) -> None:
        """Assign meaningful labels to HMM states based on their characteristics."""
        scaled = (features - self._scaler_mean) / self._scaler_std
        states = self._model.predict(scaled)

        labels = list(self._config.regime_labels)
        n_states = self._config.n_states

        # Ensure we have enough labels
        while len(labels) < n_states:
            labels.append(f"state_{len(labels)}")

        # Compute mean return and mean volatility per state
        state_stats: list[tuple[int, float, float]] = []
        for s in range(n_states):
            mask = states == s
            if np.any(mask):
                mean_ret = float(np.mean(features[mask, 0]))
                mean_vol = float(np.mean(features[mask, 1]))
                state_stats.append((s, mean_ret, mean_vol))
            else:
                state_stats.append((s, 0.0, 0.0))

        # Sort by return to assign bull/bear
        sorted_by_return = sorted(state_stats, key=lambda x: x[1])
        sorted_by_vol = sorted(state_stats, key=lambda x: x[2])

        mapping: dict[int, str] = {}
        assigned: set[str] = set()

        # Highest return state = bull
        if "bull" in labels and "bull" not in assigned:
            bull_state = sorted_by_return[-1][0]
            mapping[bull_state] = "bull"
            assigned.add("bull")

        # Lowest return state = bear
        if "bear" in labels and "bear" not in assigned:
            bear_state = sorted_by_return[0][0]
            if bear_state not in mapping:
                mapping[bear_state] = "bear"
                assigned.add("bear")

        # Highest volatility (if not already assigned) = high_volatility
        if "high_volatility" in labels and "high_volatility" not in assigned:
            for s, _, _ in reversed(sorted_by_vol):
                if s not in mapping:
                    mapping[s] = "high_volatility"
                    assigned.add("high_volatility")
                    break

        # Remaining states = ranging or other labels
        remaining_labels = [l for l in labels if l not in assigned]
        for s in range(n_states):
            if s not in mapping:
                if remaining_labels:
                    mapping[s] = remaining_labels.pop(0)
                else:
                    mapping[s] = f"state_{s}"

        self._regime_mapping = mapping

    def predict(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> RegimeState:
        """Predict the current market regime from recent observations.

        Args:
            prices: Recent price array (should include lookback window)
            volumes: Optional volume array

        Returns:
            RegimeState with current regime and probabilities.
        """
        if not self._fitted:
            raise RuntimeError("HMM not fitted. Call fit() first.")

        features = self._prepare_features(prices, volumes)
        scaled = (features - self._scaler_mean) / self._scaler_std

        # Get state probabilities for the last observation
        state_probs = self._model.predict_proba(scaled)
        last_probs = state_probs[-1]

        best_state = int(np.argmax(last_probs))
        best_label = self._regime_mapping.get(best_state, f"state_{best_state}")

        all_probs = {
            self._regime_mapping.get(i, f"state_{i}"): round(float(last_probs[i]), 4)
            for i in range(len(last_probs))
        }

        return RegimeState(
            regime_id=best_state,
            regime_label=best_label,
            probability=round(float(last_probs[best_state]), 4),
            all_probabilities=all_probs,
        )

    def predict_sequence(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> list[RegimeState]:
        """Predict regime for every timestep in the price series."""
        if not self._fitted:
            raise RuntimeError("HMM not fitted. Call fit() first.")

        features = self._prepare_features(prices, volumes)
        scaled = (features - self._scaler_mean) / self._scaler_std

        state_probs = self._model.predict_proba(scaled)
        states = self._model.predict(scaled)

        results: list[RegimeState] = []
        for i in range(len(states)):
            probs = state_probs[i]
            sid = int(states[i])
            label = self._regime_mapping.get(sid, f"state_{sid}")

            all_p = {
                self._regime_mapping.get(j, f"state_{j}"): round(float(probs[j]), 4)
                for j in range(len(probs))
            }

            results.append(RegimeState(
                regime_id=sid,
                regime_label=label,
                probability=round(float(probs[sid]), 4),
                all_probabilities=all_p,
            ))

        return results

    def detect_volatility_clustering(
        self,
        prices: np.ndarray,
        window: int = 20,
    ) -> dict[str, float]:
        """Detect GARCH-like volatility clustering in the price series.

        Returns autocorrelation of squared returns as a clustering measure.
        """
        log_prices = np.log(np.maximum(prices, 1e-10))
        returns = np.diff(log_prices)
        squared_returns = returns ** 2

        if len(squared_returns) < window * 2:
            return {"clustering_score": 0.0, "autocorrelation_lag1": 0.0}

        # Autocorrelation of squared returns at lag 1
        mean_sq = np.mean(squared_returns)
        var_sq = np.var(squared_returns)

        if var_sq < 1e-15:
            return {"clustering_score": 0.0, "autocorrelation_lag1": 0.0}

        n = len(squared_returns)
        acf_1 = float(
            np.sum((squared_returns[1:] - mean_sq) * (squared_returns[:-1] - mean_sq))
            / (n * var_sq)
        )

        # Higher ACF(1) of squared returns indicates stronger volatility clustering
        clustering_score = max(acf_1, 0.0)

        return {
            "clustering_score": round(clustering_score, 4),
            "autocorrelation_lag1": round(acf_1, 4),
        }


class _SimpleFallbackHMM:
    """Simple fallback HMM implementation when hmmlearn is not available.

    Uses K-means style initialization and basic Gaussian emission model.
    Not as accurate as hmmlearn but sufficient for regime detection.
    """

    def __init__(self, n_states: int = 4, seed: int = 42) -> None:
        self._n_states = n_states
        self._seed = seed
        self._means: np.ndarray | None = None
        self._covars: list[np.ndarray] = []
        self._transition: np.ndarray | None = None
        self._initial: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Fit using simple K-means + covariance estimation."""
        rng = np.random.RandomState(self._seed)
        n_samples, n_features = X.shape

        # K-means initialization
        indices = rng.choice(n_samples, self._n_states, replace=False)
        centroids = X[indices].copy()

        for _ in range(50):
            # Assign clusters
            dists = np.array([
                np.sum((X - c) ** 2, axis=1) for c in centroids
            ])
            labels = np.argmin(dists, axis=0)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self._n_states):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        self._means = centroids

        # Compute covariances
        self._covars = []
        for k in range(self._n_states):
            mask = labels == k
            if np.sum(mask) > 1:
                cluster_data = X[mask]
                cov = np.cov(cluster_data.T)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                self._covars.append(cov + np.eye(n_features) * 1e-6)
            else:
                self._covars.append(np.eye(n_features))

        # Estimate transition matrix from label sequence
        self._transition = np.ones((self._n_states, self._n_states)) * 0.01
        for i in range(len(labels) - 1):
            self._transition[labels[i], labels[i + 1]] += 1
        row_sums = self._transition.sum(axis=1, keepdims=True)
        self._transition = self._transition / np.where(row_sums > 0, row_sums, 1)

        # Initial state distribution
        self._initial = np.zeros(self._n_states)
        for k in range(self._n_states):
            self._initial[k] = np.sum(labels == k)
        self._initial = self._initial / self._initial.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely state for each observation."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute state probabilities for each observation."""
        n_samples = len(X)
        log_likelihoods = np.zeros((n_samples, self._n_states))

        for k in range(self._n_states):
            mean = self._means[k]
            cov = self._covars[k]
            try:
                cov_inv = np.linalg.inv(cov)
                cov_det = np.linalg.det(cov)
                if cov_det <= 0:
                    cov_det = 1e-10
            except np.linalg.LinAlgError:
                cov_inv = np.eye(cov.shape[0])
                cov_det = 1.0

            diff = X - mean
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            log_norm = -0.5 * (X.shape[1] * np.log(2 * np.pi) + np.log(cov_det))
            log_likelihoods[:, k] = exponent + log_norm

        # Add log prior
        log_likelihoods += np.log(np.clip(self._initial, 1e-10, None))

        # Softmax to get probabilities
        max_ll = np.max(log_likelihoods, axis=1, keepdims=True)
        exp_ll = np.exp(log_likelihoods - max_ll)
        probs = exp_ll / np.sum(exp_ll, axis=1, keepdims=True)

        return probs

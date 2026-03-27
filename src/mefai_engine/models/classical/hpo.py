"""Optuna hyperparameter optimization for gradient boosting models.

Integrates with walk-forward cross validation to find optimal
XGBoost and LightGBM parameters. The objective function maximizes
Sharpe ratio on the validation set of each walk-forward fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 100
    timeout_seconds: int = 3600
    model_type: Literal["xgboost", "lightgbm"] = "xgboost"
    n_folds: int = 5
    train_size: int = 3000
    val_size: int = 1000
    study_name: str = "mefai_hpo"
    direction: str = "maximize"
    seed: int = 42


class GradientBoostHPO:
    """Optuna based HPO pipeline for XGBoost and LightGBM.

    Uses walk-forward cross validation within each trial to evaluate
    parameter combinations. The objective maximizes Sharpe ratio on
    the out-of-sample validation windows.
    """

    def __init__(self, config: HPOConfig | None = None) -> None:
        self._config = config or HPOConfig()
        self._best_params: dict[str, Any] = {}
        self._study: Any = None

    @property
    def best_params(self) -> dict[str, Any]:
        return self._best_params

    def optimize(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        prices: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            features: Training feature matrix (n_samples x n_features)
            targets: Target labels (direction classes or returns)
            prices: Optional price array for Sharpe calculation

        Returns:
            Dictionary of best hyperparameters found.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("optuna is required for HPO. Run: pip install optuna")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._features = features
        self._targets = targets
        self._prices = prices

        self._study = optuna.create_study(
            study_name=self._config.study_name,
            direction=self._config.direction,
        )

        self._study.optimize(
            self._objective,
            n_trials=self._config.n_trials,
            timeout=self._config.timeout_seconds,
            show_progress_bar=False,
        )

        self._best_params = self._study.best_params
        logger.info(
            "hpo.complete",
            best_value=f"{self._study.best_value:.4f}",
            n_trials=len(self._study.trials),
            best_params=self._best_params,
        )
        return self._best_params

    def _objective(self, trial: Any) -> float:
        """Optuna objective function: walk-forward Sharpe ratio."""
        params = self._suggest_params(trial)

        # Walk-forward cross validation
        fold_sharpes: list[float] = []
        total_length = len(self._features)
        window = self._config.train_size + self._config.val_size
        step = self._config.val_size

        fold = 0
        start = 0
        while start + window <= total_length and fold < self._config.n_folds:
            train_end = start + self._config.train_size
            val_end = train_end + self._config.val_size

            X_train = self._features[start:train_end]
            y_train = self._targets[start:train_end]
            X_val = self._features[train_end:val_end]
            y_val = self._targets[train_end:val_end]

            sharpe = self._train_and_evaluate(params, X_train, y_train, X_val, y_val)
            fold_sharpes.append(sharpe)

            start += step
            fold += 1

        if not fold_sharpes:
            return -10.0

        return float(np.mean(fold_sharpes))

    def _suggest_params(self, trial: Any) -> dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        if self._config.model_type == "lightgbm":
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 5, 100)
            params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 10)

        return params

    def _train_and_evaluate(
        self,
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train a model with given params and return validation Sharpe ratio."""
        if self._config.model_type == "xgboost":
            return self._train_xgboost(params, X_train, y_train, X_val, y_val)
        return self._train_lightgbm(params, X_train, y_train, X_val, y_val)

    @staticmethod
    def _train_xgboost(
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train XGBoost and compute Sharpe on validation predictions."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required. Run: pip install xgboost")

        # Filter out LightGBM specific params
        xgb_params = {
            k: v for k, v in params.items()
            if k not in ("num_leaves", "min_data_in_leaf", "bagging_freq")
        }
        n_est = xgb_params.pop("n_estimators", 300)

        model = xgb.XGBClassifier(
            n_estimators=n_est,
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
            **xgb_params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predict probabilities and compute directional signals
        probas = model.predict_proba(X_val)
        # Classes: 0=short 1=flat 2=long
        signals = np.argmax(probas, axis=1) - 1  # Map to -1/0/1
        confidences = np.max(probas, axis=1)

        return _compute_signal_sharpe(signals, confidences, y_val)

    @staticmethod
    def _train_lightgbm(
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train LightGBM and compute Sharpe on validation predictions."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required. Run: pip install lightgbm")

        n_est = params.pop("n_estimators", 300)
        lgb_params = {k: v for k, v in params.items()}

        model = lgb.LGBMClassifier(
            n_estimators=n_est,
            objective="multiclass",
            num_class=3,
            verbosity=-1,
            random_state=42,
            **lgb_params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        probas = model.predict_proba(X_val)
        signals = np.argmax(probas, axis=1) - 1
        confidences = np.max(probas, axis=1)

        return _compute_signal_sharpe(signals, confidences, y_val)

    def get_study_dataframe(self) -> Any:
        """Return the study trials as a DataFrame (requires pandas)."""
        if self._study is None:
            return None
        return self._study.trials_dataframe()

    def get_param_importances(self) -> dict[str, float]:
        """Compute hyperparameter importance scores."""
        if self._study is None:
            return {}
        try:
            import optuna
            importances = optuna.importance.get_param_importances(self._study)
            return dict(importances)
        except Exception:
            return {}


def _compute_signal_sharpe(
    signals: np.ndarray,
    confidences: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute a simplified Sharpe ratio from signal predictions.

    Uses target direction alignment as a proxy for returns.
    """
    # Convert targets to directional returns proxy
    target_directions = targets.astype(float) - 1  # 0/1/2 -> -1/0/1
    weighted_signals = signals * confidences
    returns = weighted_signals * target_directions

    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0

    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    return sharpe

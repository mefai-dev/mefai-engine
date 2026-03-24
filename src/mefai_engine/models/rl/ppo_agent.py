"""PPO agent for position sizing decisions.

Uses Stable-Baselines3 PPO to learn optimal position sizing
based on market features and current portfolio state.
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


class PPOPositionSizer(BasePredictor):
    """PPO-based position sizing agent.

    Given market features, predicts the optimal position size [-1, 1].
    Trained using Stable-Baselines3 on historical data.
    """

    model_id = "rl_ppo"
    model_version = "v1"

    def __init__(
        self,
        total_timesteps: int = 500_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        horizon_seconds: int = 3600,
    ) -> None:
        self._total_timesteps = total_timesteps
        self._lr = learning_rate
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._gamma = gamma
        self._clip_range = clip_range
        self._horizon_seconds = horizon_seconds
        self._agent: Any = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_split: float = 0.2,
    ) -> dict[str, float]:
        """Train PPO agent on historical data.

        Args:
            features: (n_samples, n_features) feature matrix
            targets: not used directly - RL learns from environment rewards
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from mefai_engine.models.rl.env import create_gym_env

        # Use features as prices proxy for training
        # In practice, prices should be passed separately
        prices = targets if len(targets.shape) == 1 else targets[:, 0]

        split = int(len(features) * (1 - validation_split))
        train_features = features[:split]
        train_prices = prices[:split]

        env = DummyVecEnv([lambda: create_gym_env(train_features, train_prices)])

        self._agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=self._lr,
            n_steps=min(self._n_steps, len(train_features) - 2),
            batch_size=self._batch_size,
            n_epochs=self._n_epochs,
            gamma=self._gamma,
            clip_range=self._clip_range,
            verbose=0,
        )

        logger.info("ppo.training_start", timesteps=self._total_timesteps)
        self._agent.learn(total_timesteps=self._total_timesteps)
        self._trained = True

        logger.info("ppo.training_complete")
        return {"timesteps_trained": self._total_timesteps}

    def predict(self, features: np.ndarray) -> Prediction:
        """Predict position size from current features."""
        if not self._trained or self._agent is None:
            from mefai_engine.exceptions import ModelNotTrainedError
            raise ModelNotTrainedError("PPO agent not trained")

        if features.ndim == 1:
            # Add position info placeholders
            obs = np.concatenate([features, [0.0, 0.0, 0.0]]).astype(np.float32)
        else:
            obs = features[-1].astype(np.float32)

        action, _ = self._agent.predict(obs, deterministic=True)
        position_target = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # Convert continuous action to direction + size
        if position_target > 0.1:
            direction = Direction.LONG
            confidence = min(abs(position_target), 1.0)
        elif position_target < -0.1:
            direction = Direction.SHORT
            confidence = min(abs(position_target), 1.0)
        else:
            direction = Direction.FLAT
            confidence = 1.0 - abs(position_target)

        return Prediction(
            direction=direction,
            confidence=confidence,
            magnitude=abs(position_target),
            horizon_seconds=self._horizon_seconds,
            model_id=self.model_id,
            model_version=self.model_version,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def predict_batch(self, features: np.ndarray) -> list[Prediction]:
        """Not efficiently supported by SB3, predict one by one."""
        return [self.predict(features[i]) for i in range(len(features))]

    def save(self, path: Path) -> None:
        if self._agent is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._agent.save(str(path))
            logger.info("ppo.saved", path=str(path))

    def load(self, path: Path) -> None:
        from stable_baselines3 import PPO
        self._agent = PPO.load(str(path))
        self._trained = True
        logger.info("ppo.loaded", path=str(path))

"""Custom Gymnasium trading environment for RL agent training.

Simulates perpetual futures trading with:
- Realistic fees (maker/taker)
- Funding rates
- Slippage estimation
- Position sizing decisions
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


class TradingEnv:
    """Perpetual futures trading environment for RL.

    Observation: feature vector + current position info
    Action: continuous [-1, 1] where:
        -1 = max short
         0 = flat (no position)
        +1 = max long
    Reward: risk-adjusted PnL (Sharpe-like)
    """

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        funding_rates: np.ndarray | None = None,
        initial_balance: float = 10000.0,
        max_leverage: int = 10,
        maker_fee_pct: float = 0.02,
        taker_fee_pct: float = 0.05,
        slippage_pct: float = 0.01,
        episode_length: int = 0,
    ) -> None:
        self._features = features
        self._prices = prices
        self._funding = funding_rates if funding_rates is not None else np.zeros(len(prices))
        self._initial_balance = initial_balance
        self._max_leverage = max_leverage
        self._maker_fee = maker_fee_pct / 100
        self._taker_fee = taker_fee_pct / 100
        self._slippage = slippage_pct / 100
        self._episode_length = episode_length or len(prices) - 1

        n_features = features.shape[1] if features.ndim == 2 else 1
        # Observation: features + [position_size, unrealized_pnl, balance_pct]
        self.observation_dim = n_features + 3
        self.action_dim = 1  # Continuous [-1, 1]

        self._reset_state()

    def _reset_state(self) -> None:
        self._step = 0
        self._balance = self._initial_balance
        self._position = 0.0  # Positive = long, negative = short
        self._entry_price = 0.0
        self._total_fees = 0.0
        self._returns: list[float] = []
        self._equity_curve: list[float] = [self._initial_balance]
        self._peak_equity = self._initial_balance

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            start = np.random.randint(0, max(1, len(self._prices) - self._episode_length - 1))
        else:
            start = 0

        self._start_idx = start
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step.

        Args:
            action: Target position as fraction of max [-1, 1]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if isinstance(action, np.ndarray):
            target_position = float(np.clip(action[0], -1.0, 1.0))
        else:
            target_position = float(np.clip(action, -1.0, 1.0))

        idx = self._start_idx + self._step
        current_price = self._prices[idx]
        next_price = self._prices[min(idx + 1, len(self._prices) - 1)]

        # Position change
        old_position = self._position
        target_size = target_position * self._max_leverage * self._balance / current_price
        position_change = target_size - old_position

        # Fees on position change
        if abs(position_change) > 0.0001:
            fee = abs(position_change) * current_price * self._taker_fee
            slippage_cost = abs(position_change) * current_price * self._slippage
            self._total_fees += fee + slippage_cost
            self._balance -= fee + slippage_cost

        self._position = target_size
        if abs(self._position) > 0.0001 and abs(old_position) < 0.0001:
            self._entry_price = current_price

        # PnL from price change
        price_change = next_price - current_price
        pnl = self._position * price_change

        # Funding cost (every 8 hours ≈ every 480 1-minute candles)
        funding_cost = 0.0
        if abs(self._position) > 0 and idx < len(self._funding):
            funding_cost = abs(self._position) * current_price * self._funding[idx]
            self._balance -= funding_cost

        self._balance += pnl
        step_return = pnl / self._initial_balance

        self._returns.append(step_return)
        self._equity_curve.append(self._balance)

        if self._balance > self._peak_equity:
            self._peak_equity = self._balance

        self._step += 1

        # Reward: risk-adjusted return with drawdown penalty
        reward = self._compute_reward(step_return)

        # Episode termination
        terminated = self._balance <= self._initial_balance * 0.5  # 50% loss = done
        truncated = self._step >= self._episode_length

        info = {
            "balance": self._balance,
            "position": self._position,
            "pnl": pnl,
            "fees": self._total_fees,
            "step": self._step,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        idx = min(self._start_idx + self._step, len(self._features) - 1)
        features = self._features[idx]
        if features.ndim == 0:
            features = np.array([features])

        # Normalize position info
        position_norm = self._position * self._prices[idx] / self._initial_balance
        unrealized = 0.0
        if abs(self._position) > 0.0001 and self._entry_price > 0:
            unrealized = self._position * (self._prices[idx] - self._entry_price) / self._initial_balance
        balance_pct = self._balance / self._initial_balance - 1.0

        extra = np.array([position_norm, unrealized, balance_pct])
        return np.concatenate([features, extra]).astype(np.float32)

    def _compute_reward(self, step_return: float) -> float:
        """Compute reward: Sharpe-like with drawdown penalty."""
        # Base reward: step return
        reward = step_return * 100  # Scale up

        # Drawdown penalty
        if self._balance < self._peak_equity:
            dd = (self._peak_equity - self._balance) / self._peak_equity
            reward -= dd * 10

        # Fee awareness: penalize excessive trading
        if self._total_fees > self._initial_balance * 0.01:
            reward -= 0.01

        return reward


def create_gym_env(
    features: np.ndarray,
    prices: np.ndarray,
    **kwargs: Any,
) -> Any:
    """Create a Gymnasium-wrapped trading environment (if gymnasium installed)."""
    if not HAS_GYM:
        raise ImportError("gymnasium not installed. Run: pip install gymnasium")

    env = TradingEnv(features, prices, **kwargs)

    class GymTradingEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self._env = env
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._env.observation_dim,), dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
            )

        def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            return self._env.reset(seed)

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            return self._env.step(action)

    return GymTradingEnv()

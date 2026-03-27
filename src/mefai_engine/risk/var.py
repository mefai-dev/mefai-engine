"""Value at Risk (VaR) and Conditional VaR (Expected Shortfall) calculations.

Provides multiple VaR estimation methods:
1. Historical simulation
2. Parametric (normal distribution)
3. Monte Carlo simulation
4. CVaR / Expected Shortfall

Used by the risk manager to set position limits and freeze trading
when portfolio risk exceeds thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class VaRConfig:
    """Configuration for VaR calculations."""
    confidence_level: float = 0.99  # 99% VaR
    holding_period_days: int = 1
    monte_carlo_simulations: int = 10000
    lookback_days: int = 252
    max_var_pct: float = 5.0  # Max allowed VaR as pct of portfolio
    max_cvar_pct: float = 8.0  # Max allowed CVaR as pct of portfolio
    seed: int = 42


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    historical_var: float
    parametric_var: float
    monte_carlo_var: float
    cvar: float  # Conditional VaR (Expected Shortfall)
    var_pct: float  # VaR as percentage of portfolio
    cvar_pct: float  # CVaR as percentage of portfolio
    is_within_limit: bool
    breach_type: str  # "none" or "var" or "cvar" or "both"


class VaRCalculator:
    """Multi-method Value at Risk calculator.

    Computes VaR using historical simulation and parametric (normal)
    and Monte Carlo approaches. Also computes CVaR (Expected Shortfall)
    which measures the expected loss in the tail beyond VaR.
    """

    def __init__(self, config: VaRConfig | None = None) -> None:
        self._config = config or VaRConfig()
        self._returns_history: list[float] = []

    def update(self, portfolio_return: float) -> None:
        """Add a new portfolio return observation."""
        self._returns_history.append(portfolio_return)

        # Keep only lookback window
        max_keep = self._config.lookback_days * 2
        if len(self._returns_history) > max_keep:
            self._returns_history = self._returns_history[-max_keep:]

    def calculate(
        self,
        portfolio_value: float,
        returns: np.ndarray | None = None,
    ) -> VaRResult:
        """Calculate VaR using all methods.

        Args:
            portfolio_value: Current portfolio value in USDT
            returns: Optional array of returns (uses history if not provided)

        Returns:
            VaRResult with all VaR metrics.
        """
        if returns is not None:
            ret = returns
        else:
            ret = np.array(self._returns_history)

        cfg = self._config

        if len(ret) < 10:
            return VaRResult(
                historical_var=0.0,
                parametric_var=0.0,
                monte_carlo_var=0.0,
                cvar=0.0,
                var_pct=0.0,
                cvar_pct=0.0,
                is_within_limit=True,
                breach_type="none",
            )

        # Scale to holding period
        sqrt_hp = np.sqrt(cfg.holding_period_days)

        # 1. Historical VaR
        hist_var = self._historical_var(ret, cfg.confidence_level) * portfolio_value

        # 2. Parametric VaR (normal distribution)
        param_var = self._parametric_var(ret, cfg.confidence_level, sqrt_hp) * portfolio_value

        # 3. Monte Carlo VaR
        mc_var = self._monte_carlo_var(
            ret, cfg.confidence_level, cfg.monte_carlo_simulations, cfg.seed, sqrt_hp
        ) * portfolio_value

        # 4. CVaR (Expected Shortfall)
        cvar = self._conditional_var(ret, cfg.confidence_level) * portfolio_value

        # Use the most conservative (highest) VaR estimate
        max_var = max(abs(hist_var), abs(param_var), abs(mc_var))

        var_pct = (max_var / portfolio_value * 100) if portfolio_value > 0 else 0.0
        cvar_pct = (abs(cvar) / portfolio_value * 100) if portfolio_value > 0 else 0.0

        # Check limits
        var_ok = var_pct <= cfg.max_var_pct
        cvar_ok = cvar_pct <= cfg.max_cvar_pct

        if not var_ok and not cvar_ok:
            breach = "both"
        elif not var_ok:
            breach = "var"
        elif not cvar_ok:
            breach = "cvar"
        else:
            breach = "none"

        result = VaRResult(
            historical_var=round(abs(hist_var), 2),
            parametric_var=round(abs(param_var), 2),
            monte_carlo_var=round(abs(mc_var), 2),
            cvar=round(abs(cvar), 2),
            var_pct=round(var_pct, 4),
            cvar_pct=round(cvar_pct, 4),
            is_within_limit=var_ok and cvar_ok,
            breach_type=breach,
        )

        if not result.is_within_limit:
            logger.warning(
                "var.limit_breach",
                var_pct=f"{var_pct:.2f}%",
                cvar_pct=f"{cvar_pct:.2f}%",
                breach=breach,
            )

        return result

    @staticmethod
    def _historical_var(
        returns: np.ndarray,
        confidence: float,
    ) -> float:
        """Historical simulation VaR.

        Simply takes the (1 - confidence) percentile of historical returns.
        """
        percentile = (1 - confidence) * 100
        var = float(np.percentile(returns, percentile))
        return abs(var)

    @staticmethod
    def _parametric_var(
        returns: np.ndarray,
        confidence: float,
        sqrt_holding_period: float = 1.0,
    ) -> float:
        """Parametric (variance-covariance) VaR assuming normal distribution."""
        try:
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence)
        except ImportError:
            # Approximate z-scores for common confidence levels
            z_map = {0.99: -2.326, 0.95: -1.645, 0.975: -1.96}
            z_score = z_map.get(confidence, -2.326)

        mean = float(np.mean(returns))
        std = float(np.std(returns))

        var = -(mean + z_score * std) * sqrt_holding_period
        return abs(var)

    @staticmethod
    def _monte_carlo_var(
        returns: np.ndarray,
        confidence: float,
        n_simulations: int = 10000,
        seed: int = 42,
        sqrt_holding_period: float = 1.0,
    ) -> float:
        """Monte Carlo VaR simulation.

        Simulates portfolio returns by sampling from a distribution
        fitted to historical returns.
        """
        rng = np.random.RandomState(seed)
        mean = float(np.mean(returns))
        std = float(np.std(returns))

        if std < 1e-10:
            return 0.0

        # Simulate returns
        simulated = rng.normal(mean, std, size=n_simulations) * sqrt_holding_period

        percentile = (1 - confidence) * 100
        var = float(np.percentile(simulated, percentile))
        return abs(var)

    @staticmethod
    def _conditional_var(
        returns: np.ndarray,
        confidence: float,
    ) -> float:
        """Conditional VaR (Expected Shortfall / CVaR).

        The expected loss given that the loss exceeds VaR.
        CVaR is always >= VaR and provides a more complete tail risk measure.
        """
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)

        # Average of returns below VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return abs(float(var_threshold))

        cvar = float(np.mean(tail_returns))
        return abs(cvar)

    def check_position_var(
        self,
        portfolio_value: float,
        position_size_pct: float,
        position_returns: np.ndarray | None = None,
    ) -> bool:
        """Check if adding a position would keep VaR within limits.

        Args:
            portfolio_value: Current portfolio value
            position_size_pct: Proposed position size as pct of portfolio
            position_returns: Historical returns for the position's asset

        Returns:
            True if the position is acceptable from a VaR perspective.
        """
        if position_returns is None or len(position_returns) < 10:
            return True  # Cannot assess; allow but with caution

        # Scale returns by position size
        weighted_returns = position_returns * (position_size_pct / 100)

        # Combine with portfolio returns
        portfolio_ret = np.array(self._returns_history[-self._config.lookback_days:])
        if len(portfolio_ret) > 0:
            min_len = min(len(portfolio_ret), len(weighted_returns))
            combined = portfolio_ret[-min_len:] + weighted_returns[-min_len:]
        else:
            combined = weighted_returns

        result = self.calculate(portfolio_value, combined)
        return result.is_within_limit

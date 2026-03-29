"""Correlation risk matrix for multi-asset portfolio management.

Tracks rolling correlations between trading symbols and limits
aggregate correlated exposure. Uses Dynamic Conditional Correlation
(DCC) for time-varying correlation estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CorrelationConfig:
    """Configuration for correlation risk management."""
    rolling_window: int = 60  # Number of periods for rolling correlation
    btc_correlation_limit: float = 0.85  # Max allowed BTC correlation for any position
    max_correlated_exposure_pct: float = 40.0  # Max total correlated exposure
    correlation_threshold: float = 0.7  # Pairs above this are considered correlated
    dcc_alpha: float = 0.01  # DCC model alpha parameter
    dcc_beta: float = 0.95  # DCC model beta parameter
    position_reduction_factor: float = 0.5  # Reduce by this factor when limit breached


@dataclass
class CorrelationCheckResult:
    """Result of correlation risk check."""
    is_acceptable: bool
    btc_correlation: float
    max_pair_correlation: float
    total_correlated_exposure_pct: float
    suggested_size_multiplier: float
    correlated_pairs: list[tuple[str, str, float]]
    reason: str


class CorrelationRiskManager:
    """Manages correlation risk across the portfolio.

    Monitors rolling correlations between all traded symbols and
    limits total exposure to highly correlated assets. Special
    attention is given to BTC correlation since most crypto assets
    are highly correlated with BTC.
    """

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self._config = config or CorrelationConfig()
        # symbol -> list of recent returns
        self._return_history: dict[str, list[float]] = {}
        self._correlation_matrix: np.ndarray | None = None
        self._symbols: list[str] = []
        # DCC state
        self._dcc_Q: np.ndarray | None = None
        self._dcc_Q_bar: np.ndarray | None = None

    def update_returns(self, symbol: str, return_value: float) -> None:
        """Add a new return observation for a symbol.

        Args:
            symbol: Trading pair symbol
            return_value: Period return (e.g. log return)
        """
        if symbol not in self._return_history:
            self._return_history[symbol] = []
            if symbol not in self._symbols:
                self._symbols.append(symbol)

        self._return_history[symbol].append(return_value)

        # Keep only rolling window + buffer
        max_keep = self._config.rolling_window * 2
        if len(self._return_history[symbol]) > max_keep:
            self._return_history[symbol] = self._return_history[symbol][-max_keep:]

    def compute_correlation_matrix(self) -> np.ndarray:
        """Compute the rolling correlation matrix across all symbols.

        Returns:
            Correlation matrix as numpy array (n_symbols x n_symbols).
        """
        symbols = [
            s for s in self._symbols
            if len(self._return_history.get(s, [])) >= self._config.rolling_window
        ]

        if len(symbols) < 2:
            self._correlation_matrix = np.eye(len(self._symbols))
            return self._correlation_matrix

        n = len(symbols)
        window = self._config.rolling_window

        # Build return matrix
        returns_matrix = np.zeros((window, n))
        for i, sym in enumerate(symbols):
            hist = self._return_history[sym]
            returns_matrix[:, i] = hist[-window:]

        # Compute correlation matrix
        self._correlation_matrix = np.corrcoef(returns_matrix.T)
        # Fix NaN from constant columns
        self._correlation_matrix = np.nan_to_num(self._correlation_matrix, nan=0.0)
        np.fill_diagonal(self._correlation_matrix, 1.0)

        self._symbols = symbols
        return self._correlation_matrix

    def compute_dcc(self) -> np.ndarray:
        """Compute Dynamic Conditional Correlation (DCC) matrix.

        DCC captures time-varying correlations using the model:
        Q_t = (1 - alpha - beta) * Q_bar + alpha * e_{t-1} * e_{t-1}' + beta * Q_{t-1}
        R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

        Returns:
            Time-varying correlation matrix.
        """
        symbols = [
            s for s in self._symbols
            if len(self._return_history.get(s, [])) >= self._config.rolling_window
        ]

        if len(symbols) < 2:
            return np.eye(len(self._symbols))

        n = len(symbols)
        window = self._config.rolling_window
        alpha = self._config.dcc_alpha
        beta = self._config.dcc_beta

        returns_matrix = np.zeros((window, n))
        for i, sym in enumerate(symbols):
            returns_matrix[:, i] = self._return_history[sym][-window:]

        # Standardize returns
        means = np.mean(returns_matrix, axis=0)
        stds = np.std(returns_matrix, axis=0)
        stds = np.where(stds > 0, stds, 1.0)
        standardized = (returns_matrix - means) / stds

        # Unconditional correlation (Q_bar)
        Q_bar = np.corrcoef(standardized.T)
        Q_bar = np.nan_to_num(Q_bar, nan=0.0)
        np.fill_diagonal(Q_bar, 1.0)

        # Initialize DCC Q matrix
        if self._dcc_Q is None or self._dcc_Q.shape[0] != n:
            self._dcc_Q = Q_bar.copy()
            self._dcc_Q_bar = Q_bar.copy()
        else:
            self._dcc_Q_bar = Q_bar.copy()

        # DCC recursion for the latest observation
        e_t = standardized[-1:].T  # (n, 1)
        self._dcc_Q = (
            (1 - alpha - beta) * Q_bar
            + alpha * (e_t @ e_t.T)
            + beta * self._dcc_Q
        )

        # Compute R_t from Q_t
        diag_Q = np.sqrt(np.diag(self._dcc_Q))
        diag_Q = np.where(diag_Q > 0, diag_Q, 1.0)
        R_t = self._dcc_Q / np.outer(diag_Q, diag_Q)
        np.fill_diagonal(R_t, 1.0)

        self._correlation_matrix = R_t
        return R_t

    def check_correlation_risk(
        self,
        symbol: str,
        proposed_size_pct: float,
        current_positions: dict[str, float],
    ) -> CorrelationCheckResult:
        """Check if a new position would breach correlation risk limits.

        Args:
            symbol: Symbol for the proposed trade
            proposed_size_pct: Proposed position size as pct of equity
            current_positions: Dict of symbol -> current size pct

        Returns:
            CorrelationCheckResult with risk assessment.
        """
        cfg = self._config

        # Compute latest correlation matrix
        if self._correlation_matrix is None or len(self._symbols) < 2:
            self.compute_correlation_matrix()

        if self._correlation_matrix is None or len(self._symbols) < 2:
            return CorrelationCheckResult(
                is_acceptable=True,
                btc_correlation=0.0,
                max_pair_correlation=0.0,
                total_correlated_exposure_pct=0.0,
                suggested_size_multiplier=1.0,
                correlated_pairs=[],
                reason="Insufficient symbols for correlation analysis",
            )

        corr_matrix = self._correlation_matrix
        sym_idx = self._symbols.index(symbol) if symbol in self._symbols else -1

        # BTC correlation check
        btc_corr = 0.0
        btc_idx = -1
        for i, s in enumerate(self._symbols):
            if "BTC" in s.upper():
                btc_idx = i
                break

        if btc_idx >= 0 and sym_idx >= 0:
            btc_corr = abs(float(corr_matrix[sym_idx, btc_idx]))

        # Find highly correlated pairs
        correlated_pairs: list[tuple[str, str, float]] = []
        n = len(self._symbols)
        for i in range(n):
            for j in range(i + 1, n):
                c = abs(float(corr_matrix[i, j]))
                if c >= cfg.correlation_threshold:
                    correlated_pairs.append((self._symbols[i], self._symbols[j], round(c, 4)))

        # Calculate total correlated exposure
        total_corr_exposure = 0.0
        if sym_idx >= 0:
            for other_sym, size_pct in current_positions.items():
                if other_sym == symbol:
                    continue
                other_idx = self._symbols.index(other_sym) if other_sym in self._symbols else -1
                if other_idx >= 0:
                    corr = abs(float(corr_matrix[sym_idx, other_idx]))
                    if corr >= cfg.correlation_threshold:
                        total_corr_exposure += abs(size_pct) * corr

        total_corr_exposure += proposed_size_pct

        # Determine max pair correlation
        max_pair_corr = 0.0
        if sym_idx >= 0:
            for other_sym in current_positions:
                other_idx = self._symbols.index(other_sym) if other_sym in self._symbols else -1
                if other_idx >= 0 and other_idx != sym_idx:
                    c = abs(float(corr_matrix[sym_idx, other_idx]))
                    max_pair_corr = max(max_pair_corr, c)

        # Check limits
        size_multiplier = 1.0
        reasons: list[str] = []

        if btc_corr > cfg.btc_correlation_limit:
            size_multiplier = min(size_multiplier, cfg.position_reduction_factor)
            reasons.append(f"BTC correlation {btc_corr:.2f} exceeds limit {cfg.btc_correlation_limit:.2f}")

        if total_corr_exposure > cfg.max_correlated_exposure_pct:
            over_ratio = cfg.max_correlated_exposure_pct / total_corr_exposure
            size_multiplier = min(size_multiplier, over_ratio)
            reasons.append(
                f"Correlated exposure {total_corr_exposure:.1f}% "
                f"exceeds limit {cfg.max_correlated_exposure_pct:.1f}%"
            )

        is_acceptable = size_multiplier >= 0.99
        reason = " | ".join(reasons) if reasons else "Correlation risk within limits"

        if not is_acceptable:
            logger.warning(
                "correlation.risk_breach",
                symbol=symbol,
                multiplier=f"{size_multiplier:.2f}",
                reason=reason,
            )

        return CorrelationCheckResult(
            is_acceptable=is_acceptable,
            btc_correlation=round(btc_corr, 4),
            max_pair_correlation=round(max_pair_corr, 4),
            total_correlated_exposure_pct=round(total_corr_exposure, 2),
            suggested_size_multiplier=round(size_multiplier, 4),
            correlated_pairs=correlated_pairs,
            reason=reason,
        )

    def get_correlation_matrix_dict(self) -> dict[str, dict[str, float]]:
        """Return the correlation matrix as a nested dictionary."""
        if self._correlation_matrix is None:
            return {}

        result: dict[str, dict[str, float]] = {}
        for i, sym_i in enumerate(self._symbols):
            result[sym_i] = {}
            for j, sym_j in enumerate(self._symbols):
                result[sym_i][sym_j] = round(float(self._correlation_matrix[i, j]), 4)
        return result

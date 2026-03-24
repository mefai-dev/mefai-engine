"""Risk limit check functions - pure functions, no state."""

from __future__ import annotations


def check_position_size(
    requested_pct: float, max_position_pct: float
) -> tuple[bool, float]:
    """Check if position size is within limits.

    Returns:
        (is_within_limit, max_allowed_size_pct)
    """
    if requested_pct <= max_position_pct:
        return True, requested_pct
    return False, max_position_pct


def check_max_exposure(
    current_exposure_pct: float,
    new_position_pct: float,
    max_total_exposure_pct: float,
) -> tuple[bool, float]:
    """Check total portfolio exposure.

    Returns:
        (is_within_limit, remaining_capacity_pct)
    """
    remaining = max_total_exposure_pct - current_exposure_pct
    if remaining <= 0:
        return False, 0.0
    if new_position_pct <= remaining:
        return True, remaining
    return False, remaining


def check_daily_loss(
    current_daily_loss: float,
    total_equity: float,
    max_daily_loss_pct: float,
) -> bool:
    """Check if daily loss limit has been reached."""
    if total_equity <= 0:
        return False
    current_pct = (current_daily_loss / total_equity) * 100
    return current_pct < max_daily_loss_pct


def check_drawdown(
    current_drawdown_pct: float, max_drawdown_pct: float
) -> bool:
    """Check if max drawdown has been exceeded."""
    return current_drawdown_pct < max_drawdown_pct

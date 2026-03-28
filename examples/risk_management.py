"""Risk management pipeline example.

This script demonstrates how to set up and use the full MEFAI Engine
risk management stack including:
- Kelly criterion position sizing
- PnL tracking with drawdown monitoring
- Circuit breaker for automated halt
- Position and daily loss limit checks

All examples use synthetic data and require no external services.
"""

from __future__ import annotations

import numpy as np

from mefai_engine.risk.kelly import KellyCriterion, KellyConfig
from mefai_engine.risk.pnl_tracker import PnLTracker
from mefai_engine.risk.circuit_breaker import TradingCircuitBreaker
from mefai_engine.risk.limits import (
    check_daily_loss,
    check_drawdown,
    check_position_size,
)


def demo_kelly_criterion() -> None:
    """Demonstrate Kelly criterion position sizing.

    The Kelly formula calculates the optimal fraction of capital to
    risk based on historical win rate and average payoff ratio.
    Fractional Kelly (typically 0.25x) is used in practice to reduce
    the variance of returns.
    """
    print("\n" + "=" * 60)
    print("  Kelly Criterion Position Sizing")
    print("=" * 60)

    # Configure with quarter Kelly (conservative)
    config = KellyConfig(
        fraction=0.25,
        max_position_pct=10.0,
        min_win_rate=0.4,
        min_trades=30,
        confidence_scaling=True,
    )
    kelly = KellyCriterion(config)

    # Scenario 1: Strong edge
    result = kelly.calculate(
        win_rate=0.55,
        avg_winner=150.0,
        avg_loser=100.0,
        confidence=0.85,
        n_trades=200,
    )
    print(f"\n  Scenario 1 - Strong edge (55% WR / 1.5:1 payoff)")
    print(f"    Full Kelly:       {result.full_kelly_pct:.2f}%")
    print(f"    Fractional Kelly: {result.fractional_kelly_pct:.2f}%")
    print(f"    Capped Size:      {result.capped_size_pct:.2f}%")
    print(f"    Edge:             {result.edge:.4f}")
    print(f"    Valid:            {result.is_valid}")

    # Scenario 2: Weak edge
    result2 = kelly.calculate(
        win_rate=0.48,
        avg_winner=110.0,
        avg_loser=100.0,
        confidence=0.70,
        n_trades=150,
    )
    print(f"\n  Scenario 2 - Weak edge (48% WR / 1.1:1 payoff)")
    print(f"    Full Kelly:       {result2.full_kelly_pct:.2f}%")
    print(f"    Fractional Kelly: {result2.fractional_kelly_pct:.2f}%")
    print(f"    Capped Size:      {result2.capped_size_pct:.2f}%")
    print(f"    Edge:             {result2.edge:.4f}")

    # Scenario 3: From trade history
    rng = np.random.default_rng(42)
    trade_pnls = list(rng.choice([120.0, 80.0, -90.0, -60.0, 150.0, -100.0], size=100))
    result3 = kelly.calculate_from_trades(trade_pnls, confidence=0.80)
    print(f"\n  Scenario 3 - From 100 trade PnL history")
    print(f"    Full Kelly:       {result3.full_kelly_pct:.2f}%")
    print(f"    Capped Size:      {result3.capped_size_pct:.2f}%")
    print(f"    Valid:            {result3.is_valid}")
    print(f"    Reason:           {result3.reason}")


def demo_pnl_tracker() -> None:
    """Demonstrate PnL tracking and equity curve monitoring."""
    print("\n" + "=" * 60)
    print("  PnL Tracker and Equity Monitoring")
    print("=" * 60)

    tracker = PnLTracker(initial_equity=10000.0)

    # Simulate a series of trades
    trades = [120, -80, 200, -50, 150, -30, -90, 180, -60, 250, -40, 100]
    equity = 10000.0

    for pnl in trades:
        tracker.record(float(pnl))
        equity += pnl
        tracker.update_equity(equity)

    metrics = tracker.to_dict()
    print(f"\n  After {tracker.total_trades} trades:")
    print(f"    Total PnL:        ${metrics['total_pnl']:.2f}")
    print(f"    Win Rate:         {metrics['win_rate']:.1%}")
    print(f"    Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"    Expectancy:       ${metrics['expectancy']:.2f}")
    print(f"    Avg Winner:       ${metrics['avg_winner']:.2f}")
    print(f"    Avg Loser:        ${metrics['avg_loser']:.2f}")
    print(f"    Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"    Current Drawdown: {metrics['current_drawdown_pct']:.2f}%")


def demo_circuit_breaker() -> None:
    """Demonstrate the trading circuit breaker."""
    print("\n" + "=" * 60)
    print("  Circuit Breaker")
    print("=" * 60)

    cb = TradingCircuitBreaker(
        max_consecutive_losses=3,
        max_drawdown_pct=10.0,
        cooldown_seconds=0,  # Zero for demo (instant recovery)
    )

    events = ["win", "loss", "loss", "loss", "check", "win"]

    for event in events:
        if event == "win":
            cb.record_win()
            print(f"  WIN  -> State: {cb.state}")
        elif event == "loss":
            cb.record_loss()
            print(f"  LOSS -> State: {cb.state}")
        elif event == "check":
            can = cb.can_trade()
            print(f"  CHECK can_trade={can} -> State: {cb.state}")


def demo_risk_limits() -> None:
    """Demonstrate risk limit checks."""
    print("\n" + "=" * 60)
    print("  Risk Limit Checks")
    print("=" * 60)

    # Position size check
    ok, allowed = check_position_size(requested_pct=8.0, max_position_pct=10.0)
    print(f"\n  Position size 8% (max 10%): OK={ok} allowed={allowed}%")

    ok2, allowed2 = check_position_size(requested_pct=15.0, max_position_pct=10.0)
    print(f"  Position size 15% (max 10%): OK={ok2} allowed={allowed2}%")

    # Daily loss check
    dl_ok = check_daily_loss(
        current_daily_loss=250.0,
        total_equity=10000.0,
        max_daily_loss_pct=3.0,
    )
    print(f"\n  Daily loss $250 / $10000 equity (max 3%): OK={dl_ok}")

    dl_bad = check_daily_loss(
        current_daily_loss=400.0,
        total_equity=10000.0,
        max_daily_loss_pct=3.0,
    )
    print(f"  Daily loss $400 / $10000 equity (max 3%): OK={dl_bad}")

    # Drawdown check
    dd_ok = check_drawdown(current_drawdown_pct=5.0, max_drawdown_pct=10.0)
    print(f"\n  Drawdown 5% (max 10%): OK={dd_ok}")

    dd_bad = check_drawdown(current_drawdown_pct=12.0, max_drawdown_pct=10.0)
    print(f"  Drawdown 12% (max 10%): OK={dd_bad}")


def main() -> None:
    """Run all risk management demonstrations."""
    print("=" * 60)
    print("  MEFAI Engine - Risk Management Pipeline Demo")
    print("=" * 60)

    demo_kelly_criterion()
    demo_pnl_tracker()
    demo_circuit_breaker()
    demo_risk_limits()

    print("\n" + "=" * 60)
    print("  All risk management demos complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

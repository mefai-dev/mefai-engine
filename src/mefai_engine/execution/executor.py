"""Order executor - manages order lifecycle from signal to fill."""

from __future__ import annotations

import asyncio
import uuid

import structlog

from mefai_engine.constants import ExecutionAlgo, OrderType, RiskDecisionType, Side
from mefai_engine.exchange.base import BaseExchange
from mefai_engine.types import (
    Balance,
    Fill,
    OrderRequest,
    OrderResult,
    RiskDecision,
    Signal,
)

logger = structlog.get_logger()


class OrderExecutor:
    """Manages the full order lifecycle.

    Signal -> Risk check result -> Size calculation -> Order placement
    -> Fill monitoring -> Reconciliation.
    """

    def __init__(
        self,
        exchange: BaseExchange,
        maker_fee_bps: int = 2,
        taker_fee_bps: int = 5,
        max_retries: int = 3,
    ) -> None:
        self._exchange = exchange
        self._maker_fee_bps = maker_fee_bps
        self._taker_fee_bps = taker_fee_bps
        self._max_retries = max_retries
        self._active_orders: dict[str, OrderResult] = {}

    async def execute_signal(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
        balance: Balance,
        algo: ExecutionAlgo = ExecutionAlgo.MARKET,
    ) -> list[Fill]:
        """Execute a trading signal through the specified algorithm.

        Returns list of fills (may be multiple for TWAP/ICEBERG).
        """
        if risk_decision.decision == RiskDecisionType.REJECTED:
            logger.warning("executor.rejected", reason=risk_decision.reason)
            return []

        # Calculate position size in base currency
        size_pct = risk_decision.approved_size_pct
        notional = balance.available * (size_pct / 100)

        # Get current price for quantity calculation
        ticker = await self._exchange.get_ticker(signal.symbol)
        price = ticker.last
        if price <= 0:
            logger.error("executor.invalid_price", price=price)
            return []

        quantity = notional / price

        side = Side.LONG if signal.direction.value == "long" else Side.SHORT

        logger.info(
            "executor.executing",
            symbol=signal.symbol,
            side=side,
            algo=algo,
            quantity=quantity,
            notional=notional,
        )

        if algo == ExecutionAlgo.MARKET:
            return await self._execute_market(signal.symbol, side, quantity)
        elif algo == ExecutionAlgo.TWAP:
            return await self._execute_twap(signal.symbol, side, quantity)
        else:
            return await self._execute_market(signal.symbol, side, quantity)

    async def _execute_market(
        self, symbol: str, side: Side, quantity: float
    ) -> list[Fill]:
        """Direct market order execution."""
        client_id = f"mefai_{uuid.uuid4().hex[:12]}"
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            client_order_id=client_id,
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                result = await self._exchange.place_order(request)
                self._active_orders[result.order_id] = result
                logger.info(
                    "executor.order_placed",
                    order_id=result.order_id,
                    status=result.status,
                    filled=result.filled_quantity,
                    avg_price=result.average_price,
                )
                return [Fill(
                    order_id=result.order_id,
                    symbol=symbol,
                    side=side,
                    price=result.average_price,
                    quantity=result.filled_quantity,
                    fee=result.fee,
                    timestamp=result.timestamp,
                    exchange=result.exchange,
                )]
            except Exception:
                logger.exception(
                    "executor.order_failed",
                    attempt=attempt,
                    symbol=symbol,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0 * attempt)

        return []

    async def _execute_twap(
        self,
        symbol: str,
        side: Side,
        total_quantity: float,
        duration_seconds: int = 120,
        slices: int = 6,
    ) -> list[Fill]:
        """Time-Weighted Average Price execution.

        Splits order into equal slices over the duration.
        """
        slice_qty = total_quantity / slices
        interval = duration_seconds / slices
        fills: list[Fill] = []

        for i in range(slices):
            partial = await self._execute_market(symbol, side, slice_qty)
            fills.extend(partial)

            if i < slices - 1:
                await asyncio.sleep(interval)

        if fills:
            total_cost = sum(f.price * f.quantity for f in fills)
            total_qty = sum(f.quantity for f in fills)
            avg_price = total_cost / total_qty if total_qty > 0 else 0
            logger.info(
                "executor.twap_complete",
                symbol=symbol,
                total_qty=total_qty,
                avg_price=avg_price,
                slices=len(fills),
            )

        return fills

    async def cancel_all(self, symbol: str) -> int:
        """Cancel all active orders for a symbol."""
        cancelled = 0
        for order_id, order in list(self._active_orders.items()):
            if order.symbol == symbol:
                success = await self._exchange.cancel_order(order_id, symbol)
                if success:
                    cancelled += 1
                    del self._active_orders[order_id]
        return cancelled

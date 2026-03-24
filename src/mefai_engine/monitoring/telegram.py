"""Telegram bot integration for trade alerts and status."""

from __future__ import annotations

import aiohttp
import structlog

logger = structlog.get_logger()


class TelegramNotifier:
    """Sends trading notifications via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/sendMessage",
                    json={
                        "chat_id": self._chat_id,
                        "text": message,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                ) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    logger.error("telegram.send_failed", status=resp.status, body=body)
                    return False
        except Exception:
            logger.exception("telegram.error")
            return False

    async def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        leverage: int = 1,
    ) -> None:
        msg = (
            f"<b>TRADE OPENED</b>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Side: <b>{side.upper()}</b>\n"
            f"Size: <code>{quantity:.4f}</code>\n"
            f"Price: <code>{price:.2f}</code>\n"
            f"Leverage: <code>{leverage}x</code>"
        )
        await self.send(msg)

    async def notify_trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float,
        reason: str = "",
    ) -> None:
        emoji = "+" if pnl >= 0 else ""
        msg = (
            f"<b>TRADE CLOSED</b>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Side: <b>{side.upper()}</b>\n"
            f"PnL: <code>{emoji}{pnl:.2f} USDT ({emoji}{pnl_pct:.2f}%)</code>\n"
            f"Reason: {reason}"
        )
        await self.send(msg)

    async def notify_daily_report(
        self,
        equity: float,
        daily_pnl: float,
        total_trades: int,
        win_rate: float,
        drawdown_pct: float,
    ) -> None:
        emoji = "+" if daily_pnl >= 0 else ""
        msg = (
            f"<b>DAILY REPORT</b>\n"
            f"Equity: <code>{equity:.2f} USDT</code>\n"
            f"Daily PnL: <code>{emoji}{daily_pnl:.2f} USDT</code>\n"
            f"Trades: <code>{total_trades}</code>\n"
            f"Win Rate: <code>{win_rate:.1f}%</code>\n"
            f"Drawdown: <code>{drawdown_pct:.2f}%</code>"
        )
        await self.send(msg)

    async def notify_alert(self, title: str, message: str) -> None:
        msg = f"<b>{title}</b>\n{message}"
        await self.send(msg)

"""Authentication and rate limiting middleware."""

from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# API key from environment variable (required for live mode)
_API_KEY = os.environ.get("MEFAI_API_KEY", "")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting state
_rate_limits: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = 120  # requests per window


async def require_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Validate API key from X-API-Key header.

    In paper/backtest mode with no key configured this allows all requests.
    In live mode an API key is always required.
    """
    mode = os.environ.get("MEFAI__ENGINE__MODE", "paper")

    if not _API_KEY:
        if mode == "live":
            raise HTTPException(
                status_code=503,
                detail="MEFAI_API_KEY environment variable must be set in live mode",
            )
        return "anonymous"

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    if not hmac.compare_digest(api_key, _API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


async def check_rate_limit(request: Request) -> None:
    """Simple in memory rate limiter per client IP."""
    client = request.client.host if request.client else "unknown"
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_limits[client] = [t for t in _rate_limits[client] if t > cutoff]

    if len(_rate_limits[client]) >= _RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    _rate_limits[client].append(now)

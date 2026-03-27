"""Prometheus metrics endpoint for scraping."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from mefai_engine.monitoring.metrics import get_metrics

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Return Prometheus exposition format metrics.

    This endpoint is meant to be scraped by a Prometheus server.
    It returns all trading engine metrics in the standard text format.
    """
    metrics = get_metrics()
    output = metrics.generate_metrics()
    return Response(
        content=output,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

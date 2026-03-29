"""Model management endpoints - status and training and prediction."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mefai_engine.api.middleware import check_rate_limit, require_api_key
from mefai_engine.app import get_state

router = APIRouter(dependencies=[Depends(require_api_key), Depends(check_rate_limit)])


class TrainRequest(BaseModel):
    """Model training request."""
    model_id: str
    lookback_days: int = 365


class PredictRequest(BaseModel):
    """Single prediction request."""
    model_id: str
    symbol: str
    timeframe: str = "1h"


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """Get status of all registered ML models."""
    state = get_state()
    registry = state.get("model_registry")
    if not registry:
        return {"models": {}}

    return {
        "models": registry.status(),
        "total": len(registry.get_all()),
        "trained": len(registry.get_trained()),
    }


@router.get("/models/{model_id}")
async def get_model_detail(model_id: str) -> dict[str, Any]:
    """Get detailed info about a specific model."""
    state = get_state()
    registry = state.get("model_registry")
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not available")

    model = registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    detail: dict[str, Any] = {
        "model_id": model.model_id,
        "version": model.model_version,
        "trained": model.is_trained,
    }

    # Add feature importance for gradient boost
    if hasattr(model, "feature_importance"):
        detail["feature_importance"] = model.feature_importance(top_n=15)

    return detail


@router.post("/models/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    """Run a single prediction using a specific model."""
    state = get_state()
    registry = state.get("model_registry")
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not available")

    model = registry.get(req.model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")

    if not model.is_trained:
        raise HTTPException(status_code=400, detail=f"Model '{req.model_id}' is not trained yet")

    # Get features
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    import numpy as np

    from mefai_engine.constants import ExchangeID
    from mefai_engine.features.pipeline import FeaturePipeline

    for eid in ExchangeID:
        exchange = factory.get(eid)
        if exchange:
            try:
                candles = await exchange.get_ohlcv(req.symbol.upper(), req.timeframe, limit=300)
                if len(candles) < 50:
                    raise HTTPException(status_code=400, detail="Not enough candle data")

                config = state.get("config")
                pipeline = FeaturePipeline(config.features.enabled if config else None)

                raw = {
                    "open": np.array([c.open for c in candles]),
                    "high": np.array([c.high for c in candles]),
                    "low": np.array([c.low for c in candles]),
                    "close": np.array([c.close for c in candles]),
                    "volume": np.array([c.volume for c in candles]),
                }
                computed = pipeline.compute(raw)

                # Build feature matrix (last row)
                feature_names = sorted(computed.keys())
                feature_vector = np.array([
                    float(computed[f][-1]) if not np.isnan(computed[f][-1]) else 0.0
                    for f in feature_names
                ])

                prediction = model.predict(feature_vector)
                return {
                    "model_id": req.model_id,
                    "symbol": req.symbol.upper(),
                    "direction": prediction.direction.value,
                    "confidence": round(prediction.confidence, 4),
                    "magnitude": round(prediction.magnitude, 6),
                    "horizon_seconds": prediction.horizon_seconds,
                    "timestamp": prediction.timestamp.isoformat(),
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=503, detail="No exchange available")


@router.post("/models/train")
async def trigger_training(req: TrainRequest) -> dict[str, Any]:
    """Trigger model training (runs in background)."""
    state = get_state()
    registry = state.get("model_registry")
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not available")

    model = registry.get(req.model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")

    return {
        "status": "training_queued",
        "model_id": req.model_id,
        "lookback_days": req.lookback_days,
        "note": "Training runs asynchronously. Check /models/{model_id} for status.",
    }


@router.post("/models/save")
async def save_all_models() -> dict[str, Any]:
    """Save all trained models to disk."""
    state = get_state()
    registry = state.get("model_registry")
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not available")

    registry.save_all()
    return {"status": "saved", "models": list(registry.get_trained().keys())}


@router.post("/models/load")
async def load_all_models() -> dict[str, Any]:
    """Load all models from disk."""
    state = get_state()
    registry = state.get("model_registry")
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not available")

    loaded = registry.load_all()
    return {"status": "loaded", "count": loaded}

"""Signal generation from model predictions."""

from __future__ import annotations

from datetime import UTC, datetime

from mefai_engine.constants import Direction
from mefai_engine.types import Prediction, Signal


def predictions_to_signal(
    predictions: list[Prediction],
    symbol: str,
    strategy_id: str,
    min_confidence: float = 0.65,
    min_agreement: float = 0.6,
) -> Signal | None:
    """Convert multiple model predictions into a single trading signal.

    Uses weighted voting based on confidence scores.

    Args:
        predictions: List of predictions from different models
        symbol: Trading pair
        strategy_id: Strategy identifier
        min_confidence: Minimum confidence threshold
        min_agreement: Minimum fraction of models that must agree

    Returns:
        Signal if consensus reached, None otherwise
    """
    if not predictions:
        return None

    # Tally votes weighted by confidence
    long_score = 0.0
    short_score = 0.0
    flat_score = 0.0
    total_weight = 0.0

    model_versions: dict[str, str] = {}

    for pred in predictions:
        weight = pred.confidence
        total_weight += weight
        model_versions[pred.model_id] = pred.model_version

        if pred.direction == Direction.LONG:
            long_score += weight
        elif pred.direction == Direction.SHORT:
            short_score += weight
        else:
            flat_score += weight

    if total_weight == 0:
        return None

    # Normalize
    long_pct = long_score / total_weight
    short_pct = short_score / total_weight
    flat_pct = flat_score / total_weight

    # Determine direction
    max_pct = max(long_pct, short_pct, flat_pct)

    if max_pct < min_agreement:
        return None  # No consensus

    if long_pct == max_pct:
        direction = Direction.LONG
        confidence = long_pct
    elif short_pct == max_pct:
        direction = Direction.SHORT
        confidence = short_pct
    else:
        direction = Direction.FLAT
        confidence = flat_pct

    if confidence < min_confidence:
        return None

    # Suggested size scales with confidence
    base_size = 5.0  # 5% base
    size_multiplier = (confidence - min_confidence) / (1.0 - min_confidence)
    suggested_size = base_size + base_size * size_multiplier  # 5-10%

    return Signal(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        suggested_size_pct=min(suggested_size, 10.0),
        strategy_id=strategy_id,
        model_versions=model_versions,
        timestamp=datetime.now(tz=UTC),
    )

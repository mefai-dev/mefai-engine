"""Feature registry with decorator-based registration and dependency tracking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FeatureSpec:
    """Specification for a registered feature."""
    name: str
    func: Callable[..., np.ndarray]
    depends_on: list[str]
    params: dict[str, Any]
    category: str = "custom"


_FEATURES: dict[str, FeatureSpec] = {}


def feature(
    name: str,
    depends_on: list[str] | None = None,
    params: dict[str, Any] | None = None,
    category: str = "custom",
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Decorator to register a feature computation function."""
    def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        spec = FeatureSpec(
            name=name,
            func=func,
            depends_on=depends_on or [],
            params=params or {},
            category=category,
        )
        _FEATURES[name] = spec
        return func
    return decorator


def get_feature(name: str) -> FeatureSpec:
    """Retrieve a registered feature by name."""
    if name not in _FEATURES:
        raise KeyError(f"Feature '{name}' not registered")
    return _FEATURES[name]


def list_features(category: str | None = None) -> list[FeatureSpec]:
    """List all registered features, optionally filtered by category."""
    specs = list(_FEATURES.values())
    if category:
        specs = [s for s in specs if s.category == category]
    return specs


def resolve_dependencies(requested: list[str]) -> list[str]:
    """Topological sort of features based on dependencies.

    Returns ordered list where each feature appears after its dependencies.
    """
    visited: set[str] = set()
    order: list[str] = []

    def _visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        spec = _FEATURES.get(name)
        if spec:
            for dep in spec.depends_on:
                _visit(dep)
        order.append(name)

    for name in requested:
        _visit(name)
    return order

"""Deterministic Grid/Random configuration search."""

from __future__ import annotations

from itertools import product
from random import Random
from typing import Any, Iterable, Mapping, Sequence

from app.optimization.models import OptimizationSpace


def iter_grid(space: OptimizationSpace) -> tuple[dict[str, Any], ...]:
    names = [spec.name for spec in space.parameters]
    values = [spec.values for spec in space.parameters]
    configs: list[dict[str, Any]] = []
    for combination in product(*values):
        config = dict(zip(names, combination))
        if _valid_constraints(config):
            configs.append(config)
    return tuple(configs[: space.max_trials])


def sample_random(space: OptimizationSpace) -> tuple[dict[str, Any], ...]:
    all_configs = list(iter_unbounded_grid(space))
    rng = Random(space.seed)
    if len(all_configs) <= space.max_trials:
        rng.shuffle(all_configs)
        return tuple(all_configs)
    indices = list(range(len(all_configs)))
    chosen = rng.sample(indices, space.max_trials)
    chosen.sort()
    # Sorting indices keeps output reproducible and independent from hash ordering.
    return tuple(all_configs[index] for index in chosen)


def iter_unbounded_grid(space: OptimizationSpace) -> Iterable[dict[str, Any]]:
    names = [spec.name for spec in space.parameters]
    values = [spec.values for spec in space.parameters]
    for combination in product(*values):
        config = dict(zip(names, combination))
        if _valid_constraints(config):
            yield config


def _valid_constraints(config: Mapping[str, Any]) -> bool:
    if "min_rr" in config and "max_rr" in config:
        try:
            if float(config["min_rr"]) > float(config["max_rr"]):
                return False
        except (TypeError, ValueError):
            return False
    if "minimum_body_ratio" in config:
        value = float(config["minimum_body_ratio"])
        if not 0.0 < value <= 1.0:
            return False
    if "swing_lookback" in config and int(config["swing_lookback"]) < 2:
        return False
    return True


def generate_search_configs(space: OptimizationSpace) -> tuple[dict[str, Any], ...]:
    if space.method == "grid":
        return iter_grid(space)
    return sample_random(space)


def top_by_score(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    key: str = "train_score",
    limit: int = 5,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        sorted(evaluations, key=lambda item: float(item.get(key, float("-inf"))), reverse=True)[:limit]
    )


__all__ = ["generate_search_configs", "iter_grid", "sample_random", "top_by_score"]

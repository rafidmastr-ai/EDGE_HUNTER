"""Local parameter sensitivity and instability diagnostics."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence


def neighbor_configs(
    config: Mapping[str, Any],
    candidate_configs: Sequence[Mapping[str, Any]],
    *,
    max_distance: int = 1,
) -> tuple[Mapping[str, Any], ...]:
    keys = tuple(config)
    neighbors: list[Mapping[str, Any]] = []
    for candidate in candidate_configs:
        if candidate is config or dict(candidate) == dict(config):
            continue
        distance = sum(candidate.get(key) != config.get(key) for key in keys)
        if distance <= max_distance:
            neighbors.append(candidate)
    return tuple(neighbors)


def sensitivity_from_scores(
    config: Mapping[str, Any],
    score_map: Mapping[str, float],
    key_for_config,
    *,
    max_distance: int = 1,
) -> dict[str, Any]:
    base_id = key_for_config(config)
    if base_id not in score_map:
        return {
            "sensitivity_score": None,
            "neighbor_count": 0,
            "worst_neighbor_deterioration": None,
        }
    base = float(score_map[base_id])
    neighbors = []
    # key_for_config is responsible for deterministic identity.
    for candidate_id, score in score_map.items():
        candidate = candidate_id if isinstance(candidate_id, Mapping) else None
        if candidate is None:
            continue
        distance = sum(candidate.get(key) != config.get(key) for key in config)
        if 0 < distance <= max_distance:
            neighbors.append(float(score))
    if not neighbors:
        return {
            "sensitivity_score": None,
            "neighbor_count": 0,
            "worst_neighbor_deterioration": None,
        }
    deteriorations = [
        (base - score) / max(abs(base), 1e-9)
        for score in neighbors
    ]
    worst = max(deteriorations)
    return {
        "sensitivity_score": max(0.0, 1.0 - max(0.0, worst)),
        "neighbor_count": len(neighbors),
        "worst_neighbor_deterioration": worst,
    }


def compute_sensitivity(
    evaluations: Sequence[Any],
    *,
    score_attr: str = "validation_score",
    max_distance: int = 1,
) -> dict[str, dict[str, Any]]:
    """Compute local sensitivity using all evaluated configurations."""
    rows = {tuple(sorted(item.config.items())): item for item in evaluations}
    output: dict[str, dict[str, Any]] = {}
    for identity, item in rows.items():
        neighbors = []
        for other_identity, other in rows.items():
            if other_identity == identity:
                continue
            distance = _config_distance(item.config, other.config)
            if 0 < distance <= max_distance:
                neighbors.append(other)
        base_score = float(getattr(item, score_attr))
        if not neighbors:
            output[item.evaluation_id] = {
                "sensitivity_score": None,
                "neighbor_count": 0,
                "worst_neighbor_deterioration": None,
            }
            continue
        deteriorations = [
            (base_score - float(getattr(other, score_attr))) / max(abs(base_score), 1e-9)
            for other in neighbors
        ]
        worst = max(deteriorations)
        output[item.evaluation_id] = {
            "sensitivity_score": max(0.0, 1.0 - max(0.0, worst)),
            "neighbor_count": len(neighbors),
            "worst_neighbor_deterioration": worst,
        }
    return output


def _config_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    keys = set(left) | set(right)
    return sum(left.get(key) != right.get(key) for key in keys)


__all__ = ["compute_sensitivity", "neighbor_configs", "sensitivity_from_scores"]

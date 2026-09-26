"""L9 learned-policy contracts and immutable policy persistence primitives.

This module contains policy identity/versioning only.  Policies are explicit
configuration/inference contracts; they never train models, promote models, or
mutate the production strategy path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class LearnedPolicyError(ValueError):
    """Base error for L9 learned-policy validation."""


class PolicyMode(str, Enum):
    """Runtime mode for an explicit policy application."""

    DISABLED = "DISABLED"
    SHADOW = "SHADOW"
    CANDIDATE = "CANDIDATE"


class PolicyType(str, Enum):
    PARAMETER_POLICY = "PARAMETER_POLICY"
    FILTER_POLICY = "FILTER_POLICY"
    CONFIRMATION_POLICY = "CONFIRMATION_POLICY"
    ENTRY_POLICY = "ENTRY_POLICY"
    EXIT_POLICY = "EXIT_POLICY"
    MODEL_POLICY = "MODEL_POLICY"


_ALLOWED_STRATEGIES = {"CLASSIC", "SMC", "ICT", "AI"}
_SENSITIVE_PARTS = ("password", "api_key", "secret", "session", "token", "subscription")
_WEIGHT_KEYS = {
    "weights",
    "strategy_weights",
    "classic_weight",
    "smc_weight",
    "ict_weight",
    "ai_weight",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LearnedPolicyError("policy payload contains a non-finite float")
        return value
    return value


def _validate_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LearnedPolicyError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LearnedPolicyError(f"{path} contains a non-string key")
            lowered = key.lower()
            if any(part in lowered for part in _SENSITIVE_PARTS):
                raise LearnedPolicyError(f"{path} contains a prohibited sensitive field: {key}")
            if lowered in _WEIGHT_KEYS or lowered.endswith("_weights"):
                raise LearnedPolicyError("fixed strategy-weight policies are not permitted")
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    raise LearnedPolicyError(f"{path} contains unsupported value type: {type(value).__name__}")


def _sha256(payload: Mapping[str, Any]) -> str:
    body = json.dumps(_canonical(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LearnedPolicyError(f"{name} must be timezone-aware")


@dataclass(frozen=True)
class LearnedPolicyVersion:
    """Immutable strategy-policy version with deterministic identity and lineage."""

    policy_version_id: str
    policy_version_label: str
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    policy_types: tuple[PolicyType, ...]
    source_model_version_id: str | None
    source_training_run_id: str | None
    source_dataset_version_id: str | None
    feature_schema_version: str | None
    label_version: str | None
    policy_parameters: Mapping[str, Any]
    enabled_capabilities: tuple[str, ...]
    fingerprint: str
    created_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)
    default_mode: PolicyMode = PolicyMode.DISABLED

    @classmethod
    def create(
        cls,
        *,
        policy_version_label: str,
        strategy_name: str,
        strategy_variant: str,
        strategy_version: str,
        policy_types: tuple[PolicyType, ...] | list[PolicyType],
        source_model_version_id: str | None = None,
        source_training_run_id: str | None = None,
        source_dataset_version_id: str | None = None,
        feature_schema_version: str | None = None,
        label_version: str | None = None,
        policy_parameters: Mapping[str, Any] | None = None,
        enabled_capabilities: tuple[str, ...] | list[str] = (),
        metadata: Mapping[str, Any] | None = None,
        default_mode: PolicyMode = PolicyMode.DISABLED,
        created_at: datetime | None = None,
    ) -> "LearnedPolicyVersion":
        now = created_at or datetime.now(timezone.utc)
        _aware(now, "created_at")
        normalized_types = tuple(dict.fromkeys(PolicyType(item) for item in policy_types))
        normalized_capabilities = tuple(sorted({str(item).strip() for item in enabled_capabilities if str(item).strip()}))
        parameters = dict(policy_parameters or {})
        meta = dict(metadata or {})
        payload = {
            "policy_version_label": policy_version_label,
            "strategy_name": strategy_name,
            "strategy_variant": strategy_variant,
            "strategy_version": strategy_version,
            "policy_types": [item.value for item in normalized_types],
            "source_model_version_id": source_model_version_id,
            "source_training_run_id": source_training_run_id,
            "source_dataset_version_id": source_dataset_version_id,
            "feature_schema_version": feature_schema_version,
            "label_version": label_version,
            "policy_parameters": parameters,
            "enabled_capabilities": list(normalized_capabilities),
            "metadata": meta,
            "default_mode": PolicyMode(default_mode).value,
        }
        fingerprint = _sha256(payload)
        return cls(
            policy_version_id=f"pol-{fingerprint[:32]}",
            policy_version_label=policy_version_label,
            strategy_name=strategy_name,
            strategy_variant=strategy_variant,
            strategy_version=strategy_version,
            policy_types=normalized_types,
            source_model_version_id=source_model_version_id,
            source_training_run_id=source_training_run_id,
            source_dataset_version_id=source_dataset_version_id,
            feature_schema_version=feature_schema_version,
            label_version=label_version,
            policy_parameters=parameters,
            enabled_capabilities=normalized_capabilities,
            fingerprint=fingerprint,
            created_at=now,
            metadata=meta,
            default_mode=PolicyMode(default_mode),
        )

    def __post_init__(self) -> None:
        for name in (
            "policy_version_id",
            "policy_version_label",
            "strategy_name",
            "strategy_variant",
            "strategy_version",
            "fingerprint",
        ):
            if not str(getattr(self, name)).strip():
                raise LearnedPolicyError(f"{name} must not be empty")
        _aware(self.created_at, "created_at")
        if self.strategy_name.upper() not in _ALLOWED_STRATEGIES:
            raise LearnedPolicyError(f"unsupported learned-policy strategy: {self.strategy_name}")
        if not self.policy_types:
            raise LearnedPolicyError("policy_types must contain at least one PolicyType")
        if self.default_mode not in {PolicyMode.DISABLED, PolicyMode.SHADOW, PolicyMode.CANDIDATE}:
            raise LearnedPolicyError("L9 policy mode cannot be production-like")
        if self.source_model_version_id is None and PolicyType.MODEL_POLICY in self.policy_types and self.strategy_name.upper() == "AI":
            raise LearnedPolicyError("AI MODEL_POLICY requires source_model_version_id")
        _validate_json(self.policy_parameters, "policy_parameters")
        _validate_json(self.metadata, "metadata")

        expected_payload = {
            "policy_version_label": self.policy_version_label,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "policy_types": [item.value for item in self.policy_types],
            "source_model_version_id": self.source_model_version_id,
            "source_training_run_id": self.source_training_run_id,
            "source_dataset_version_id": self.source_dataset_version_id,
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
            "policy_parameters": self.policy_parameters,
            "enabled_capabilities": list(self.enabled_capabilities),
            "metadata": self.metadata,
            "default_mode": self.default_mode.value,
        }
        _validate_runtime_policy_parameters(self.policy_parameters)
        if _sha256(expected_payload) != self.fingerprint:
            raise LearnedPolicyError("policy fingerprint does not match immutable policy contents")
        expected_id = f"pol-{self.fingerprint[:32]}"
        if self.policy_version_id != expected_id:
            raise LearnedPolicyError("policy_version_id does not match policy fingerprint")

    def lineage(self) -> dict[str, str | None]:
        return {
            "policy_version_id": self.policy_version_id,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "source_model_version_id": self.source_model_version_id,
            "source_training_run_id": self.source_training_run_id,
            "source_dataset_version_id": self.source_dataset_version_id,
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.lineage(),
            "policy_version_label": self.policy_version_label,
            "policy_types": [item.value for item in self.policy_types],
            "policy_parameters": _canonical(self.policy_parameters),
            "enabled_capabilities": list(self.enabled_capabilities),
            "fingerprint": self.fingerprint,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "metadata": _canonical(self.metadata),
            "default_mode": self.default_mode.value,
        }


def _validate_runtime_policy_parameters(params: Mapping[str, Any]) -> None:
    """Validate only explicit L9 runtime-policy constraints; never optimize them."""
    overrides = params.get("parameter_overrides", {})
    if overrides and not isinstance(overrides, Mapping):
        raise LearnedPolicyError("parameter_overrides must be a mapping")
    effective = dict(overrides) if isinstance(overrides, Mapping) else {}
    if "min_rr" in effective and float(effective["min_rr"]) < 1.5:
        raise LearnedPolicyError("min_rr cannot be below 1.5")
    if "max_rr" in effective and float(effective["max_rr"]) > 2.0:
        raise LearnedPolicyError("max_rr cannot exceed 2.0")
    if "min_rr" in effective and "max_rr" in effective and float(effective["min_rr"]) > float(effective["max_rr"]):
        raise LearnedPolicyError("policy min_rr must be <= max_rr")
    if "target_rr" in params and not 1.5 <= float(params["target_rr"]) <= 2.0:
        raise LearnedPolicyError("target_rr must remain within 1.5..2.0")
    if "entry_offset_risk_fraction" in params and not -0.5 <= float(params["entry_offset_risk_fraction"]) <= 0.5:
        raise LearnedPolicyError("entry_offset_risk_fraction must be in [-0.5, 0.5]")
    if "stop_distance_multiplier" in params and not 0.25 <= float(params["stop_distance_multiplier"]) <= 2.0:
        raise LearnedPolicyError("stop_distance_multiplier must be in [0.25, 2.0]")
    if "risk_percent" in params or "lot_size" in params:
        raise LearnedPolicyError("learned policy cannot override risk percent or lot size")


__all__ = ["LearnedPolicyError", "LearnedPolicyVersion", "PolicyMode", "PolicyType"]

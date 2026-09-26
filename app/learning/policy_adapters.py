"""L9 deterministic learned-policy adapters for Classic/SMC/ICT and AI inference.

Adapters wrap existing strategy/model contracts; they do not rewrite strategy
implementations and are never invoked automatically by the production API path.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from app.learning.feature_store import FeatureSnapshot
from app.learning.policies import LearnedPolicyError, LearnedPolicyVersion, PolicyMode, PolicyType
from app.learning.training import LearningTrainingPipeline
from app.learning.model_registry_repository import ModelRegistryRepository
from app.learning.registry import ModelRegistryError, ModelVersionStatus
from app.strategies.models import SignalDirection, SignalState, Strategy, StrategyConfig, StrategyContext, StrategySignal

logger = logging.getLogger("edge_hunter.learning.policy")


class LearnedPolicyApplicationError(LearnedPolicyError):
    """Raised when an explicit learned-policy application is unsafe."""


@dataclass(frozen=True)
class ShadowPolicyResult:
    """Comparison output where the base signal remains authoritative."""

    base_signal: StrategySignal
    learned_signal: StrategySignal
    final_signal: StrategySignal
    policy_version_id: str
    application_mode: PolicyMode
    differences: Mapping[str, Any]
    audit_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AIInferenceResult:
    """AI artifact inference output; model score is not final confidence."""

    predicted_label: str
    class_scores: Mapping[str, float]
    model_version_id: str
    policy_version_id: str
    strategy_name: str
    strategy_version: str
    feature_schema_version: str
    label_version: str
    audit_metadata: Mapping[str, Any]


class _StrategyFactory(Protocol):
    name: str
    variant: str
    config: StrategyConfig

    def generate(self, context: StrategyContext) -> StrategySignal: ...


class _BaseStrategyAdapter:
    SUPPORTED_PARAMETER_KEYS = frozenset(
        {"min_rr", "max_rr", "swing_lookback", "minimum_body_ratio", "minimum_confirmation_score"}
    )
    MIN_RR = 1.5
    MAX_RR = 2.0

    def __init__(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name

    def validate(self, policy: LearnedPolicyVersion, strategy: _StrategyFactory) -> None:
        if policy.strategy_name.upper() != self.strategy_name.upper():
            raise LearnedPolicyApplicationError(
                f"policy strategy mismatch: policy={policy.strategy_name}, strategy={strategy.name}"
            )
        if policy.strategy_variant != strategy.variant:
            raise LearnedPolicyApplicationError(
                f"policy variant mismatch: policy={policy.strategy_variant}, strategy={strategy.variant}"
            )
        if policy.strategy_version != strategy.variant:
            raise LearnedPolicyApplicationError(
                f"policy strategy_version mismatch: policy={policy.strategy_version}, strategy={strategy.variant}"
            )
        self._validate_parameters(policy.policy_parameters)

    def build_strategy(self, strategy: _StrategyFactory, policy: LearnedPolicyVersion) -> _StrategyFactory:
        overrides = dict(policy.policy_parameters.get("parameter_overrides", {}))
        unknown = sorted(set(overrides) - self.SUPPORTED_PARAMETER_KEYS)
        if unknown:
            raise LearnedPolicyApplicationError(f"unsupported strategy parameter(s): {unknown}")
        config = replace(strategy.config, **overrides) if overrides else strategy.config
        return type(strategy)(config)

    def _validate_parameters(self, params: Mapping[str, Any]) -> None:
        forbidden = {"strategy_weights", "classic_weight", "smc_weight", "ict_weight", "ai_weight", "weights"}
        lowered = {str(key).lower() for key in params}
        if lowered & forbidden or any(key.endswith("_weights") for key in lowered):
            raise LearnedPolicyApplicationError("fixed strategy weights are not permitted")
        overrides = params.get("parameter_overrides", {})
        if overrides and not isinstance(overrides, Mapping):
            raise LearnedPolicyApplicationError("parameter_overrides must be a mapping")
        if "min_rr" in overrides and float(overrides["min_rr"]) < self.MIN_RR:
            raise LearnedPolicyApplicationError("min_rr cannot be below the project risk/reward floor 1.5")
        if "max_rr" in overrides and float(overrides["max_rr"]) > self.MAX_RR:
            raise LearnedPolicyApplicationError("max_rr cannot exceed the project risk/reward ceiling 2.0")
        if "min_rr" in overrides and "max_rr" in overrides and float(overrides["min_rr"]) > float(overrides["max_rr"]):
            raise LearnedPolicyApplicationError("policy min_rr must be <= max_rr")
        if "swing_lookback" in overrides and int(overrides["swing_lookback"]) < 2:
            raise LearnedPolicyApplicationError("swing_lookback must be >= 2")
        if "minimum_body_ratio" in overrides:
            value = float(overrides["minimum_body_ratio"])
            if not 0.0 < value <= 1.0:
                raise LearnedPolicyApplicationError("minimum_body_ratio must be in (0, 1]")
        if "minimum_confirmation_score" in overrides and float(overrides["minimum_confirmation_score"]) < 0:
            raise LearnedPolicyApplicationError("minimum_confirmation_score must be >= 0")
        if "risk_percent" in params or "lot_size" in params:
            raise LearnedPolicyApplicationError("learned policy cannot override risk percent or lot size")


class ClassicLearnedPolicyAdapter(_BaseStrategyAdapter):
    def __init__(self) -> None:
        super().__init__("Classic")


class SMCLearnedPolicyAdapter(_BaseStrategyAdapter):
    def __init__(self) -> None:
        super().__init__("SMC")


class ICTLearnedPolicyAdapter(_BaseStrategyAdapter):
    def __init__(self) -> None:
        super().__init__("ICT")


class LearnedStrategyAdapter:
    """Explicit policy wrapper for existing rule-based strategy implementations."""

    _ADAPTERS = {
        "CLASSIC": ClassicLearnedPolicyAdapter(),
        "SMC": SMCLearnedPolicyAdapter(),
        "ICT": ICTLearnedPolicyAdapter(),
    }

    def apply(
        self,
        strategy: _StrategyFactory,
        context: StrategyContext,
        policy: LearnedPolicyVersion | None,
        *,
        mode: PolicyMode | None = None,
    ) -> ShadowPolicyResult:
        base_signal = strategy.generate(context)
        requested_mode = mode or (policy.default_mode if policy is not None else PolicyMode.DISABLED)
        if requested_mode == PolicyMode.DISABLED:
            return _result_for_disabled(base_signal, strategy, policy)
        if policy is None:
            logger.info("Learned policy unavailable; base strategy remains unchanged")
            return _result_for_unavailable(base_signal, strategy, requested_mode)
        if strategy.name.upper() not in self._ADAPTERS:
            raise LearnedPolicyApplicationError(
                f"strategy adapter is not available for {strategy.name!r}; AI uses AIModelInferenceAdapter"
            )
        adapter = self._ADAPTERS[strategy.name.upper()]
        adapter.validate(policy, strategy)
        learned_strategy = adapter.build_strategy(strategy, policy)
        learned_signal = learned_strategy.generate(context)
        learned_signal = _apply_entry_exit_policies(learned_signal, policy)
        learned_signal = _apply_filter_confirmation_policies(learned_signal, policy)
        learned_signal = _with_policy_metadata(learned_signal, policy)
        differences = _signal_differences(base_signal, learned_signal)
        audit = {
            "policy_version_id": policy.policy_version_id,
            "strategy_name": strategy.name,
            "strategy_variant": strategy.variant,
            "strategy_version": strategy.variant,
            "application_mode": requested_mode.value,
            "source_model_version_id": policy.source_model_version_id,
            "source_training_run_id": policy.source_training_run_id,
            "source_dataset_version_id": policy.source_dataset_version_id,
            "feature_schema_version": policy.feature_schema_version,
            "label_version": policy.label_version,
            "validation_result": "VALID",
        }
        logger.info("Learned policy application", extra={"policy_audit": audit})
        final = base_signal if requested_mode == PolicyMode.SHADOW else learned_signal
        return ShadowPolicyResult(
            base_signal=base_signal,
            learned_signal=learned_signal,
            final_signal=final,
            policy_version_id=policy.policy_version_id,
            application_mode=requested_mode,
            differences=differences,
            audit_metadata=audit,
        )


def _apply_entry_exit_policies(signal: StrategySignal, policy: LearnedPolicyVersion) -> StrategySignal:
    if signal.state != SignalState.SIGNAL:
        return signal
    params = policy.policy_parameters
    entry_offset = params.get("entry_offset_risk_fraction")
    target_rr = params.get("target_rr")
    stop_multiplier = params.get("stop_distance_multiplier")
    entry = signal.entry
    stop = signal.stop_loss
    target = signal.target
    if entry is None or stop is None or target is None:
        return signal
    risk = abs(float(entry) - float(stop))
    if risk <= 0:
        return _neutralize(signal, "learned_policy_invalid_risk_geometry")
    if entry_offset is not None:
        offset = float(entry_offset)
        if not -0.5 <= offset <= 0.5:
            raise LearnedPolicyApplicationError("entry_offset_risk_fraction must be in [-0.5, 0.5]")
        signed = risk * offset
        entry = float(entry) + signed
        risk = abs(float(entry) - float(stop))
    if stop_multiplier is not None:
        multiplier = float(stop_multiplier)
        if not 0.25 <= multiplier <= 2.0:
            raise LearnedPolicyApplicationError("stop_distance_multiplier must be in [0.25, 2.0]")
        new_risk = risk * multiplier
        if signal.direction == SignalDirection.BUY:
            stop = float(entry) - new_risk
        else:
            stop = float(entry) + new_risk
        risk = new_risk
    if target_rr is not None:
        rr = float(target_rr)
        if not 1.5 <= rr <= 2.0:
            raise LearnedPolicyApplicationError("target_rr must remain within the project R:R range 1.5..2.0")
        if signal.direction == SignalDirection.BUY:
            target = float(entry) + risk * rr
        else:
            target = float(entry) - risk * rr
    elif entry_offset is not None or stop_multiplier is not None:
        rr = float(signal.risk_reward or 0.0)
        if rr <= 0:
            return _neutralize(signal, "learned_policy_invalid_rr")
        if not 1.5 <= rr <= 2.0:
            rr = min(2.0, max(1.5, rr))
        if signal.direction == SignalDirection.BUY:
            target = float(entry) + risk * rr
        else:
            target = float(entry) - risk * rr

    final_risk = abs(float(entry) - float(stop))
    final_reward = abs(float(target) - float(entry))
    final_rr = final_reward / final_risk if final_risk > 0 else 0.0
    if not 1.5 <= final_rr <= 2.0:
        raise LearnedPolicyApplicationError("learned policy produced R:R outside 1.5..2.0")
    if signal.direction == SignalDirection.BUY:
        valid = float(stop) < float(entry) < float(target)
    else:
        valid = float(target) < float(entry) < float(stop)
    if not valid:
        return _neutralize(signal, "learned_policy_invalid_price_geometry")
    return replace(signal, entry=float(entry), stop_loss=float(stop), target=float(target), risk_reward=round(final_rr, 6))


def _apply_filter_confirmation_policies(signal: StrategySignal, policy: LearnedPolicyVersion) -> StrategySignal:
    if signal.state != SignalState.SIGNAL:
        return signal
    params = policy.policy_parameters
    required_evidence = params.get("required_evidence", ())
    if required_evidence:
        wanted = tuple(str(item).strip().lower() for item in required_evidence if str(item).strip())
        available = {item.lower() for item in signal.evidence}
        if not set(wanted).issubset(available):
            return _neutralize(signal, "learned_policy_required_evidence_missing")
    minimum_score_sum = params.get("minimum_score_input_sum")
    if minimum_score_sum is not None:
        threshold = float(minimum_score_sum)
        if sum(float(value) for value in signal.score_inputs.values()) < threshold:
            return _neutralize(signal, "learned_policy_confirmation_threshold_not_met")
    return signal


def _neutralize(signal: StrategySignal, reason: str) -> StrategySignal:
    return StrategySignal(
        timestamp=signal.timestamp,
        symbol=signal.symbol,
        timeframe=signal.timeframe,
        direction=SignalDirection.NO_SIGNAL,
        state=SignalState.NO_SIGNAL,
        entry=None,
        stop_loss=None,
        target=None,
        risk_reward=None,
        entry_logic="Learned policy rejected the base signal.",
        invalidation=signal.invalidation,
        stop_loss_logic="No active stop-loss after policy rejection.",
        target_logic="No active target after policy rejection.",
        evidence=tuple(signal.evidence) + (reason,),
        score_inputs=dict(signal.score_inputs),
        strategy_name=signal.strategy_name,
        variant=signal.variant,
        metadata={**dict(signal.metadata), "learned_policy_rejection": reason},
    )


def _with_policy_metadata(signal: StrategySignal, policy: LearnedPolicyVersion) -> StrategySignal:
    return replace(
        signal,
        metadata={
            **dict(signal.metadata),
            "learned_policy_version": policy.policy_version_id,
            "learned_policy_fingerprint": policy.fingerprint,
            "learned_policy_source_model_version": policy.source_model_version_id,
            "learned_policy_source_dataset_version": policy.source_dataset_version_id,
            "learned_policy_mode": policy.default_mode.value,
        },
    )


def _signal_differences(base: StrategySignal, learned: StrategySignal) -> dict[str, Any]:
    return {
        "state_changed": base.state.value != learned.state.value,
        "direction_changed": base.direction.value != learned.direction.value,
        "entry_changed": base.entry != learned.entry,
        "stop_loss_changed": base.stop_loss != learned.stop_loss,
        "target_changed": base.target != learned.target,
        "risk_reward_changed": base.risk_reward != learned.risk_reward,
        "base_state": base.state.value,
        "learned_state": learned.state.value,
    }


def _result_for_disabled(base: StrategySignal, strategy: _StrategyFactory, policy: LearnedPolicyVersion | None) -> ShadowPolicyResult:
    return ShadowPolicyResult(
        base_signal=base,
        learned_signal=base,
        final_signal=base,
        policy_version_id=policy.policy_version_id if policy else "NONE",
        application_mode=PolicyMode.DISABLED,
        differences={"policy_disabled": True},
        audit_metadata={
            "policy_version_id": policy.policy_version_id if policy else None,
            "strategy_name": strategy.name,
            "strategy_variant": strategy.variant,
            "application_mode": PolicyMode.DISABLED.value,
            "validation_result": "NOT_APPLIED",
        },
    )


def _result_for_unavailable(base: StrategySignal, strategy: _StrategyFactory, mode: PolicyMode) -> ShadowPolicyResult:
    return ShadowPolicyResult(
        base_signal=base,
        learned_signal=base,
        final_signal=base,
        policy_version_id="NONE",
        application_mode=mode,
        differences={"policy_unavailable": True},
        audit_metadata={
            "policy_version_id": None,
            "strategy_name": strategy.name,
            "strategy_variant": strategy.variant,
            "application_mode": mode.value,
            "fallback": "BASE_STRATEGY_UNCHANGED",
            "validation_result": "NOT_AVAILABLE",
        },
    )


class AIModelInferenceAdapter:
    """Inference-only adapter for a registered L6 AI model artifact."""

    FORBIDDEN_FEATURE_KEYS = {
        "outcome",
        "exit_price",
        "exit_timestamp",
        "exit_reason",
        "duration",
        "duration_seconds",
        "r_multiple",
        "pnl",
        "future_high",
        "future_low",
        "future_volatility",
        "future_regime",
    }

    def __init__(self, registry_repository: ModelRegistryRepository, training_pipeline: LearningTrainingPipeline) -> None:
        self.registry_repository = registry_repository
        self.training_pipeline = training_pipeline

    def infer(self, policy: LearnedPolicyVersion, snapshot: FeatureSnapshot) -> AIInferenceResult:
        if policy.strategy_name.upper() != "AI":
            raise LearnedPolicyApplicationError("AIModelInferenceAdapter requires an AI policy")
        if policy.source_model_version_id is None:
            raise LearnedPolicyApplicationError("AI policy has no registered ModelVersion lineage")
        model_version = self.registry_repository.get(policy.source_model_version_id)
        if model_version is None:
            raise LearnedPolicyApplicationError("registered ModelVersion not found")
        if model_version.status == ModelVersionStatus.REVOKED:
            raise LearnedPolicyApplicationError("revoked ModelVersion cannot be used for inference")
        if model_version.strategy_name.upper() != "AI" or model_version.strategy_variant != policy.strategy_variant or model_version.strategy_version != policy.strategy_version:
            raise LearnedPolicyApplicationError("AI policy and ModelVersion strategy lineage mismatch")
        if policy.feature_schema_version is not None and model_version.feature_schema_version != policy.feature_schema_version:
            raise LearnedPolicyApplicationError("AI policy feature schema mismatch")
        if policy.label_version is not None and model_version.label_version != policy.label_version:
            raise LearnedPolicyApplicationError("AI policy label version mismatch")
        if snapshot.feature_schema_version != model_version.feature_schema_version:
            raise LearnedPolicyApplicationError("FeatureSnapshot feature schema does not match registered ModelVersion")
        if snapshot.symbol.strip() == "" or snapshot.timeframe.strip() == "":
            raise LearnedPolicyApplicationError("FeatureSnapshot identity is incomplete")
        for key in snapshot.features.keys():
            lowered = str(key).lower()
            if lowered in self.FORBIDDEN_FEATURE_KEYS or any(part in lowered for part in ("api_key", "password", "secret", "session", "token", "subscription")):
                raise LearnedPolicyApplicationError(f"forbidden inference feature: {key}")

        artifact = self.training_pipeline.load_artifact(model_version.artifact_path, expected_sha256=model_version.artifact_sha256)
        if artifact.get("strategy_name") != model_version.strategy_name or artifact.get("strategy_variant") != model_version.strategy_variant or artifact.get("strategy_version") != model_version.strategy_version:
            raise LearnedPolicyApplicationError("registered artifact strategy lineage mismatch")
        if artifact.get("feature_schema_version") != model_version.feature_schema_version or artifact.get("label_version") != model_version.label_version:
            raise LearnedPolicyApplicationError("registered artifact schema lineage mismatch")
        feature_names = tuple(str(item) for item in artifact.get("feature_names", ()))
        if set(snapshot.features.keys()) != set(feature_names):
            raise LearnedPolicyApplicationError("FeatureSnapshot feature names do not match the registered artifact")
        raw = [[snapshot.features[name] for name in feature_names]]
        transformed = artifact["preprocessor"].transform(raw)
        predicted = artifact["model"].predict(transformed)
        label_encoder = artifact["label_encoder"]
        predicted_label = str(label_encoder.inverse_transform(predicted)[0])
        class_scores: dict[str, float] = {}
        if hasattr(artifact["model"], "predict_proba"):
            probabilities = artifact["model"].predict_proba(transformed)[0]
            classes = tuple(str(item) for item in label_encoder.classes_)
            class_scores = {label: float(score) for label, score in zip(classes, probabilities)}
        audit = {
            "policy_version_id": policy.policy_version_id,
            "model_version_id": model_version.model_version_id,
            "feature_schema_version": model_version.feature_schema_version,
            "label_version": model_version.label_version,
            "inference_mode": "TRANSFORM_ONLY",
            "fit_called": False,
            "retrain_called": False,
        }
        logger.info("AI learned policy inference", extra={"policy_audit": audit})
        return AIInferenceResult(
            predicted_label=predicted_label,
            class_scores=class_scores,
            model_version_id=model_version.model_version_id,
            policy_version_id=policy.policy_version_id,
            strategy_name=model_version.strategy_name,
            strategy_version=model_version.strategy_version,
            feature_schema_version=model_version.feature_schema_version,
            label_version=model_version.label_version,
            audit_metadata=audit,
        )


class LearnedPolicyService:
    """Deterministic inference-time loader and compatibility gate for L9 policies.

    The service is an explicit integration boundary. It loads an immutable policy
    by ID, verifies any ModelVersion lineage, and then delegates to the existing
    strategy/model adapters. It is never called by the web/API decision path.
    """

    def __init__(
        self,
        policy_repository: Any,
        model_registry_repository: ModelRegistryRepository,
        training_pipeline: LearningTrainingPipeline,
    ) -> None:
        self.policy_repository = policy_repository
        self.model_registry_repository = model_registry_repository
        self.training_pipeline = training_pipeline
        self.strategy_adapter = LearnedStrategyAdapter()
        self.ai_adapter = AIModelInferenceAdapter(model_registry_repository, training_pipeline)

    def load_policy(self, policy_version_id: str) -> LearnedPolicyVersion:
        """Load an immutable policy by ID without implicitly applying it.

        Compatibility and source-lineage validation happens at the explicit
        application/inference boundary, so DISABLED mode can remain an exact
        no-op for the existing strategy path.
        """
        policy = self.policy_repository.get(policy_version_id)
        if policy is None:
            raise LearnedPolicyApplicationError(f"LearnedPolicyVersion not found: {policy_version_id}")
        return policy

    def validate_lineage(
        self,
        policy: LearnedPolicyVersion,
        *,
        strategy: _StrategyFactory | None = None,
        context: StrategyContext | None = None,
    ) -> None:
        if policy.source_model_version_id is not None:
            model_version = self.model_registry_repository.get(policy.source_model_version_id)
            if model_version is None:
                raise LearnedPolicyApplicationError("policy source ModelVersion was not found")
            if model_version.status == ModelVersionStatus.REVOKED:
                raise LearnedPolicyApplicationError("policy source ModelVersion is revoked")
            pairs = {
                "strategy_name": (policy.strategy_name, model_version.strategy_name),
                "strategy_variant": (policy.strategy_variant, model_version.strategy_variant),
                "strategy_version": (policy.strategy_version, model_version.strategy_version),
                "feature_schema_version": (policy.feature_schema_version, model_version.feature_schema_version),
                "label_version": (policy.label_version, model_version.label_version),
            }
            for field_name, (expected, actual) in pairs.items():
                if expected is not None and str(expected) != str(actual):
                    raise LearnedPolicyApplicationError(
                        f"policy/model lineage mismatch: {field_name}={expected!r} != {actual!r}"
                    )
            if policy.source_training_run_id is not None and policy.source_training_run_id != model_version.training_run_id:
                raise LearnedPolicyApplicationError("policy source TrainingRun lineage mismatch")
            if policy.source_dataset_version_id is not None and policy.source_dataset_version_id != model_version.dataset_version_id:
                raise LearnedPolicyApplicationError("policy source DatasetVersion lineage mismatch")
        elif policy.strategy_name.upper() == "AI" and policy.default_mode != PolicyMode.DISABLED:
            raise LearnedPolicyApplicationError("active AI policy requires a registered ModelVersion")

        if strategy is not None:
            if strategy.name.upper() != policy.strategy_name.upper():
                raise LearnedPolicyApplicationError(
                    f"policy strategy mismatch: policy={policy.strategy_name}, strategy={strategy.name}"
                )
            if strategy.name.upper() == "AI":
                raise LearnedPolicyApplicationError(
                    "AI inference must use AIModelInferenceAdapter, not a rule-based strategy adapter"
                )
            adapter = self.strategy_adapter._ADAPTERS.get(strategy.name.upper())
            if adapter is None:
                raise LearnedPolicyApplicationError(f"no learned-policy adapter for strategy {strategy.name!r}")
            adapter.validate(policy, strategy)
            if context is not None and policy.feature_schema_version is not None:
                metadata = context.current.metadata
                context_schema = metadata.get("feature_schema_version") if isinstance(metadata, Mapping) else None
                if context_schema is None:
                    raise LearnedPolicyApplicationError(
                        "strategy context does not expose feature_schema_version required by policy"
                    )
                if str(context_schema) != policy.feature_schema_version:
                    raise LearnedPolicyApplicationError(
                        "strategy context feature schema does not match learned policy"
                    )

    def apply_strategy(
        self,
        strategy: _StrategyFactory,
        context: StrategyContext,
        policy_version_id: str,
        *,
        mode: PolicyMode | None = None,
    ) -> ShadowPolicyResult:
        policy = self.load_policy(policy_version_id)
        requested_mode = mode or policy.default_mode
        if requested_mode == PolicyMode.DISABLED:
            # Disabled mode is deliberately a no-op even when lineage metadata is
            # stale; existing strategy behavior is preserved exactly.
            return self.strategy_adapter.apply(strategy, context, policy, mode=PolicyMode.DISABLED)
        self.validate_lineage(policy, strategy=strategy, context=context)
        return self.strategy_adapter.apply(strategy, context, policy, mode=requested_mode)

    def infer_ai(self, policy_version_id: str, snapshot: FeatureSnapshot) -> AIInferenceResult:
        policy = self.load_policy(policy_version_id)
        if policy.strategy_name.upper() != "AI":
            raise LearnedPolicyApplicationError("AI inference requires an AI learned policy")
        self.validate_lineage(policy)
        return self.ai_adapter.infer(policy, snapshot)


__all__ = [
    "AIInferenceResult",
    "AIModelInferenceAdapter",
    "ClassicLearnedPolicyAdapter",
    "ICTLearnedPolicyAdapter",
    "LearnedPolicyApplicationError",
    "LearnedPolicyService",
    "LearnedStrategyAdapter",
    "SMCLearnedPolicyAdapter",
    "ShadowPolicyResult",
]

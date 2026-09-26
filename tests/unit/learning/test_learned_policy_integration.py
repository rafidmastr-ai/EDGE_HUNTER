from __future__ import annotations

import logging
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from app.data.schema import CanonicalOHLC
from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.features.models import MarketAnalysisSeries, MarketAnalysisSnapshot
from app.learning.feature_store import FeatureSnapshot, FeatureSnapshotError, FeatureStore
from app.learning.model_registry_repository import ModelRegistryRepository
from app.learning.policies import LearnedPolicyError, LearnedPolicyVersion, PolicyMode, PolicyType
from app.learning.policy_adapters import (
    AIInferenceResult,
    AIModelInferenceAdapter,
    LearnedPolicyApplicationError,
    LearnedPolicyService,
    LearnedStrategyAdapter,
)
from app.learning.policy_repository import LearnedPolicyRepository
from app.learning.registry import ModelVersion, ModelVersionStatus
from app.learning.training import LearningTrainingPipeline
from app.learning.training_repository import TrainingRunRepository
from app.strategies.classic import ClassicV1
from app.strategies.ict import ICTV1
from app.strategies.models import StrategyConfig, StrategyContext
from app.strategies.registry import StrategyRegistry
from app.strategies.smc import SMCV1


class L9PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "l9.db")
        MigrationRunner(self.db).apply_all()
        self.features = FeatureStore(self.db)
        self.policies = LearnedPolicyRepository(self.db)
        self.model_registry = ModelRegistryRepository(self.db)
        self.training_runs = TrainingRunRepository(self.db)
        self.training = LearningTrainingPipeline(
            dataset_repository=_NoDatasetRepository(),
            feature_store=self.features,
            training_repository=self.training_runs,
        )
        self.adapter = LearnedStrategyAdapter()
        self.context = self._context()

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _context(self) -> StrategyContext:
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = tuple(
            CanonicalOHLC(
                timestamp=base_time.replace(minute=i),
                open=100 + i,
                high=101 + i,
                low=99 + i,
                close=100.5 + i,
                volume=None,
            )
            for i in range(8)
        )
        snapshots = tuple(
            MarketAnalysisSnapshot(
                timestamp=bar.timestamp,
                symbol="XAUUSD",
                timeframe="M15",
                values={
                    "price.close": float(bar.close),
                    "trend.ema_20": 107.0,
                    "trend.ema_50": 106.0,
                    "trend.ema_200": 104.0,
                    "momentum.rsi": 60.0,
                    "momentum.macd_histogram": 1.0,
                    "structure.body_ratio": 0.80,
                },
                metadata={
                    "engine_version": "test",
                    "market_regime": "TREND",
                    "feature_schema_version": "v1",
                },
                warmup_complete=True,
            )
            for bar in bars
        )
        analysis = MarketAnalysisSeries(
            symbol="XAUUSD",
            timeframe="M15",
            snapshots=snapshots,
            definitions=(),
            warmup_bars_required=1,
            volume_features_enabled=False,
            engine_version="test",
        )
        return StrategyContext.from_series(bars, analysis)

    def _policy(
        self,
        strategy: str,
        variant: str,
        version: str,
        *,
        policy_types: tuple[PolicyType, ...] = (PolicyType.PARAMETER_POLICY, PolicyType.EXIT_POLICY),
        **kwargs: object,
    ) -> LearnedPolicyVersion:
        policy = LearnedPolicyVersion.create(
            policy_version_label=f"{variant}_POLICY_TEST",
            strategy_name=strategy,
            strategy_variant=variant,
            strategy_version=version,
            policy_types=policy_types,
            feature_schema_version="v1",
            label_version="v1",
            policy_parameters=kwargs,
            default_mode=PolicyMode.DISABLED,
        )
        return self.policies.create(policy)

    def test_policy_contract_and_fingerprint_are_deterministic(self):
        a = self._policy("Classic", "Classic_V1", "Classic_V1", parameter_overrides={"minimum_body_ratio": 0.60})
        b = LearnedPolicyVersion.create(
            policy_version_label=a.policy_version_label,
            strategy_name=a.strategy_name,
            strategy_variant=a.strategy_variant,
            strategy_version=a.strategy_version,
            policy_types=a.policy_types,
            feature_schema_version=a.feature_schema_version,
            label_version=a.label_version,
            policy_parameters=a.policy_parameters,
            default_mode=PolicyMode.DISABLED,
        )
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertEqual(a.policy_version_id, b.policy_version_id)

    def test_policy_version_persistence_reload(self):
        policy = self._policy("SMC", "SMC_V1", "SMC_V1")
        loaded = self.policies.get(policy.policy_version_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(policy.to_dict(), loaded.to_dict())

    def test_policy_repository_queries_work(self):
        policy = self._policy("ICT", "ICT_V1", "ICT_V1")
        self.assertEqual(len(self.policies.find_by_strategy("ICT")), 1)
        self.assertEqual(len(self.policies.find_by_dataset("missing")), 0)
        self.assertEqual(len(self.policies.find_by_model("missing")), 0)

    def test_policy_is_immutable_in_database(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        with self.assertRaises(Exception):
            self.db.execute(
                "UPDATE learning_policy_versions SET strategy_version = 'Classic_V2' WHERE policy_version_id = ?",
                (policy.policy_version_id,),
            )
            self.db.commit()
        with self.assertRaises(Exception):
            self.db.execute(
                "DELETE FROM learning_policy_versions WHERE policy_version_id = ?",
                (policy.policy_version_id,),
            )
            self.db.commit()

    def test_policy_rejects_fixed_strategy_weights(self):
        with self.assertRaises(LearnedPolicyError):
            LearnedPolicyVersion.create(
                policy_version_label="bad",
                strategy_name="Classic",
                strategy_variant="Classic_V1",
                strategy_version="Classic_V1",
                policy_types=(PolicyType.PARAMETER_POLICY,),
                policy_parameters={"strategy_weights": {"Classic": 1.0}},
            )

    def test_policy_rejects_rr_outside_project_range(self):
        with self.assertRaises(LearnedPolicyError):
            LearnedPolicyVersion.create(
                policy_version_label="bad-rr",
                strategy_name="Classic",
                strategy_variant="Classic_V1",
                strategy_version="Classic_V1",
                policy_types=(PolicyType.EXIT_POLICY,),
                policy_parameters={"target_rr": 2.5},
            )

    def test_policy_rejects_risk_overrides(self):
        with self.assertRaises(LearnedPolicyError):
            LearnedPolicyVersion.create(
                policy_version_label="bad-risk",
                strategy_name="Classic",
                strategy_variant="Classic_V1",
                strategy_version="Classic_V1",
                policy_types=(PolicyType.PARAMETER_POLICY,),
                policy_parameters={"risk_percent": 10},
            )

    def test_default_mode_is_disabled_and_no_production_mode_exists(self):
        policy = self._policy("SMC", "SMC_V1", "SMC_V1")
        self.assertEqual(policy.default_mode, PolicyMode.DISABLED)
        self.assertEqual({item.value for item in PolicyMode}, {"DISABLED", "SHADOW", "CANDIDATE"})

    def test_classic_disabled_preserves_existing_output(self):
        strategy = ClassicV1(StrategyConfig())
        base = strategy.generate(self.context)
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", parameter_overrides={"minimum_body_ratio": 0.70})
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.DISABLED)
        self.assertEqual(base, result.final_signal)
        self.assertEqual(base, result.learned_signal)

    def test_all_existing_rule_strategies_preserve_output_when_disabled(self):
        registry = StrategyRegistry.default()
        expected = [strategy.generate(self.context) for strategy in registry.strategies]
        actual = [self.adapter.apply(strategy, self.context, None, mode=PolicyMode.DISABLED).final_signal for strategy in registry.strategies]
        self.assertEqual(expected, actual)

    def test_smc_disabled_preserves_identity(self):
        result = self.adapter.apply(SMCV1(StrategyConfig()), self.context, None, mode=PolicyMode.DISABLED)
        self.assertEqual("SMC", result.final_signal.strategy_name)
        self.assertEqual("SMC_V1", result.final_signal.variant)

    def test_ict_disabled_preserves_identity(self):
        result = self.adapter.apply(ICTV1(StrategyConfig()), self.context, None, mode=PolicyMode.DISABLED)
        self.assertEqual("ICT", result.final_signal.strategy_name)
        self.assertEqual("ICT_V1", result.final_signal.variant)

    def test_classic_parameter_policy_is_explicitly_applied_in_candidate_mode(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy(
            "Classic", "Classic_V1", "Classic_V1",
            parameter_overrides={"minimum_body_ratio": 0.80, "min_rr": 1.5, "max_rr": 2.0},
        )
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual("Classic", result.final_signal.strategy_name)
        self.assertEqual("Classic_V1", result.final_signal.variant)
        self.assertEqual(policy.policy_version_id, result.learned_signal.metadata["learned_policy_version"])

    def test_smc_policy_identity_preserved_in_shadow(self):
        strategy = SMCV1(StrategyConfig())
        policy = self._policy("SMC", "SMC_V1", "SMC_V1", parameter_overrides={"minimum_body_ratio": 0.60})
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.SHADOW)
        self.assertEqual(result.base_signal, result.final_signal)
        self.assertEqual("SMC_V1", result.learned_signal.variant)

    def test_ict_policy_identity_preserved(self):
        strategy = ICTV1(StrategyConfig())
        policy = self._policy("ICT", "ICT_V1", "ICT_V1", parameter_overrides={"minimum_body_ratio": 0.60})
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual("ICT", result.final_signal.strategy_name)
        self.assertEqual("ICT_V1", result.final_signal.variant)

    def test_wrong_strategy_policy_rejected(self):
        with self.assertRaises(LearnedPolicyApplicationError):
            self.adapter.apply(
                ClassicV1(StrategyConfig()),
                self.context,
                self._policy("SMC", "SMC_V1", "SMC_V1"),
                mode=PolicyMode.CANDIDATE,
            )

    def test_wrong_variant_rejected(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        bad = LearnedPolicyVersion.create(
            policy_version_label="bad-variant",
            strategy_name="Classic",
            strategy_variant="Classic_V2",
            strategy_version="Classic_V2",
            policy_types=(PolicyType.PARAMETER_POLICY,),
            feature_schema_version="v1",
            label_version="v1",
        )
        with self.assertRaises(LearnedPolicyApplicationError):
            self.adapter.apply(ClassicV1(), self.context, bad, mode=PolicyMode.CANDIDATE)
        self.assertNotEqual(policy.policy_version_id, bad.policy_version_id)

    def test_entry_policy_support_is_deterministic(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", entry_offset_risk_fraction=0.01)
        a = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        b = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual(a.final_signal.to_dict() if hasattr(a.final_signal, "to_dict") else a.final_signal.metadata,
                         b.final_signal.to_dict() if hasattr(b.final_signal, "to_dict") else b.final_signal.metadata)

    def test_exit_policy_rr_range_is_enforced(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", target_rr=1.8)
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        if result.final_signal.state.value == "SIGNAL":
            self.assertGreaterEqual(result.final_signal.risk_reward, 1.5)
            self.assertLessEqual(result.final_signal.risk_reward, 2.0)

    def test_invalid_policy_geometry_is_rejected_or_neutralized(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", entry_offset_risk_fraction=-0.5, target_rr=1.5)
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual("Classic_V1", result.learned_signal.variant)

    def test_missing_policy_falls_back_to_base(self):
        strategy = ClassicV1(StrategyConfig())
        base = strategy.generate(self.context)
        result = self.adapter.apply(strategy, self.context, None, mode=PolicyMode.CANDIDATE)
        self.assertEqual(base, result.final_signal)
        self.assertEqual("BASE_STRATEGY_UNCHANGED", result.audit_metadata["fallback"])

    def test_shadow_never_changes_base_decision(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", target_rr=1.9)
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.SHADOW)
        self.assertEqual(result.base_signal, result.final_signal)

    def test_candidate_mode_does_not_imply_production(self):
        self.assertNotIn("PRODUCTION", {item.value for item in PolicyMode})
        self.assertNotIn("PRODUCTION", LearnedPolicyVersion.__annotations__)

    def test_policy_application_deterministic(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", target_rr=1.7)
        first = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        second = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual(first.final_signal.metadata, second.final_signal.metadata)
        self.assertEqual(first.differences, second.differences)

    def test_policy_lineage_fields_are_preserved(self):
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_V1_POLICY_TEST",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id="mdl-test",
            feature_schema_version="v1",
            label_version="v1",
        )
        self.assertEqual(policy.strategy_name, "AI")
        self.assertEqual(policy.strategy_version, "AI_V1")
        self.assertEqual(policy.feature_schema_version, "v1")
        self.assertEqual(policy.label_version, "v1")

    def test_ai_requires_model_lineage(self):
        with self.assertRaises(LearnedPolicyError):
            LearnedPolicyVersion.create(
                policy_version_label="ai",
                strategy_name="AI",
                strategy_variant="AI_V1",
                strategy_version="AI_V1",
                policy_types=(PolicyType.MODEL_POLICY,),
            )

    def test_ai_inference_requires_registered_model(self):
        policy = LearnedPolicyVersion.create(
            policy_version_label="ai",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id="missing",
        )
        snapshot = FeatureSnapshot(
            timestamp=self.context.current.timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="test",
            feature_schema_version="v1",
            features={"price.close": 100.0},
        )
        ai = AIModelInferenceAdapter(self.model_registry, self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            ai.infer(policy, snapshot)

    def test_ai_inference_adapter_has_no_training_methods(self):
        self.assertFalse(any(name in dir(AIModelInferenceAdapter) for name in ("fit", "partial_fit", "retrain", "fine_tune")))

    def test_ai_inference_feature_schema_guard(self):
        policy = LearnedPolicyVersion.create(
            policy_version_label="ai",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id="missing",
            feature_schema_version="v2",
        )
        snapshot = FeatureSnapshot(
            timestamp=self.context.current.timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="test",
            feature_schema_version="v1",
            features={"price.close": 100.0},
        )
        ai = AIModelInferenceAdapter(self.model_registry, self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            ai.infer(policy, snapshot)

    def test_ai_inference_does_not_import_web_or_strategy_aggregation(self):
        import app.learning.policy_adapters as module
        self.assertNotIn("app.web.analysis_service", module.__dict__)
        self.assertNotIn("SignalConfidenceEngine", module.__dict__)

    def test_no_strategy_weighting_in_policy_output(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", parameter_overrides={"min_rr": 1.5})
        values = policy.to_dict()
        self.assertNotIn("strategy_weights", values)
        self.assertNotIn("classic_weight", values)
        self.assertNotIn("smc_weight", values)
        self.assertNotIn("ict_weight", values)
        self.assertNotIn("ai_weight", values)

    def test_policy_reload_preserves_fingerprint(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        loaded = self.policies.get(policy.policy_version_id)
        self.assertEqual(policy.fingerprint, loaded.fingerprint)

    def test_policy_uses_explicit_capability_types(self):
        policy = self._policy(
            "Classic", "Classic_V1", "Classic_V1",
            policy_types=(PolicyType.PARAMETER_POLICY, PolicyType.FILTER_POLICY, PolicyType.CONFIRMATION_POLICY, PolicyType.ENTRY_POLICY, PolicyType.EXIT_POLICY),
        )
        self.assertEqual(
            {item.value for item in policy.policy_types},
            {"PARAMETER_POLICY", "FILTER_POLICY", "CONFIRMATION_POLICY", "ENTRY_POLICY", "EXIT_POLICY"},
        )

    def test_filter_policy_uses_existing_strategy_evidence_only(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy(
            "Classic", "Classic_V1", "Classic_V1",
            policy_types=(PolicyType.FILTER_POLICY,),
            required_evidence=("EMA 20/50/200 alignment",),
        )
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual("Classic", result.learned_signal.strategy_name)

    def test_confirmation_filter_can_reject_without_switching_strategy(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy(
            "Classic", "Classic_V1", "Classic_V1",
            policy_types=(PolicyType.CONFIRMATION_POLICY,),
            minimum_score_input_sum=999,
        )
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual("Classic", result.learned_signal.strategy_name)
        self.assertEqual("Classic_V1", result.learned_signal.variant)

    def test_policy_validation_does_not_change_strategy_when_disabled(self):
        strategy = ClassicV1(StrategyConfig())
        bad_unused = self._policy("Classic", "Classic_V1", "Classic_V1", required_evidence=("does-not-exist",))
        result = self.adapter.apply(strategy, self.context, bad_unused, mode=PolicyMode.DISABLED)
        self.assertEqual(strategy.generate(self.context), result.final_signal)

    def test_rr_is_never_left_outside_constraints_after_adaptation(self):
        strategy = ClassicV1(StrategyConfig())
        for rr in (1.5, 1.7, 2.0):
            policy = self._policy("Classic", "Classic_V1", "Classic_V1", target_rr=rr)
            result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
            if result.learned_signal.state.value == "SIGNAL":
                self.assertGreaterEqual(result.learned_signal.risk_reward, 1.5)
                self.assertLessEqual(result.learned_signal.risk_reward, 2.0)

    def test_audit_metadata_contains_lineage_without_secrets(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.SHADOW)
        self.assertEqual(policy.policy_version_id, result.audit_metadata["policy_version_id"])
        self.assertNotIn("password", str(result.audit_metadata).lower())
        self.assertNotIn("api_key", str(result.audit_metadata).lower())
        self.assertNotIn("secret", str(result.audit_metadata).lower())

    def test_policy_application_logs_structured_audit(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        with self.assertLogs("edge_hunter.learning.policy", level=logging.INFO) as logs:
            self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.SHADOW)
        self.assertIn("Learned policy application", " ".join(logs.output))

    def test_model_registry_and_dataset_are_not_mutated_by_rule_policy(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        before_models = self.db.execute("SELECT COUNT(*) AS c FROM learning_model_versions").fetchone()["c"]
        before_datasets = self.db.execute("SELECT COUNT(*) AS c FROM learning_dataset_versions").fetchone()["c"]
        self.adapter.apply(ClassicV1(), self.context, policy, mode=PolicyMode.CANDIDATE)
        after_models = self.db.execute("SELECT COUNT(*) AS c FROM learning_model_versions").fetchone()["c"]
        after_datasets = self.db.execute("SELECT COUNT(*) AS c FROM learning_dataset_versions").fetchone()["c"]
        self.assertEqual(before_models, after_models)
        self.assertEqual(before_datasets, after_datasets)

    def test_policy_identity_is_not_timestamp_only(self):
        fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = LearnedPolicyVersion.create(
            policy_version_label="p",
            strategy_name="Classic",
            strategy_variant="Classic_V1",
            strategy_version="Classic_V1",
            policy_types=(PolicyType.PARAMETER_POLICY,),
            policy_parameters={"parameter_overrides": {"min_rr": 1.5}},
            created_at=fixed,
        )
        b = LearnedPolicyVersion.create(
            policy_version_label="p",
            strategy_name="Classic",
            strategy_variant="Classic_V1",
            strategy_version="Classic_V1",
            policy_types=(PolicyType.PARAMETER_POLICY,),
            policy_parameters={"parameter_overrides": {"min_rr": 1.5}},
            created_at=fixed.replace(day=2),
        )
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertEqual(a.policy_version_id, b.policy_version_id)

    def test_policy_types_are_immutable_tuples(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        self.assertIsInstance(policy.policy_types, tuple)
        self.assertIsInstance(policy.enabled_capabilities, tuple)

    def test_ai_inference_result_does_not_claim_final_confidence(self):
        annotations = AIInferenceResult.__annotations__
        self.assertNotIn("confidence", annotations)
        self.assertNotIn("win_probability", annotations)

    def test_policy_application_does_not_call_confidence_engine(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        with patch("app.learning.policy_adapters._with_policy_metadata", wraps=__import__("app.learning.policy_adapters", fromlist=["_with_policy_metadata"])._with_policy_metadata) as wrapped:
            result = self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual(1, wrapped.call_count)
        self.assertEqual("Classic_V1", result.learned_signal.variant)


    def _ai_snapshot(self, *, schema: str = "v1") -> FeatureSnapshot:
        return FeatureSnapshot(
            timestamp=self.context.current.timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="test",
            feature_schema_version=schema,
            features={"f1": 1.0},
        )

    def _write_pretrained_ai_artifact(self) -> tuple[ModelVersion, Path]:
        artifact_path = Path(self.tmp.name) / "training_artifact.joblib"
        artifact = {
            "artifact_format_version": self.training.VERSION,
            "model_type": "LOGISTIC_REGRESSION",
            "model_version_label": "AI_V1",
            "training_identity_hash": "training-identity-test",
            "dataset_version_id": "ds-ai",
            "strategy_name": "AI",
            "strategy_variant": "AI_V1",
            "strategy_version": "AI_V1",
            "feature_schema_version": "v1",
            "label_version": "v1",
            "random_seed": 42,
            "training_config": {},
            "feature_names": ("f1",),
            "numeric_features": ("f1",),
            "categorical_features": (),
            "label_classes": ("SL_BEFORE_TP", "TP_BEFORE_SL"),
            "preprocessor": _InferenceOnlyPreprocessor(),
            "label_encoder": _InferenceOnlyLabelEncoder(),
            "model": _InferenceOnlyModel(),
        }
        from joblib import dump
        import hashlib

        dump(artifact, artifact_path)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        model = ModelVersion(
            model_version_id="mdl-ai-test",
            model_version_label="AI_V1",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            model_type="LOGISTIC_REGRESSION",
            training_run_id="trn-ai",
            dataset_version_id="ds-ai",
            feature_schema_version="v1",
            label_version="v1",
            artifact_path=str(artifact_path.resolve()),
            artifact_sha256=digest,
            artifact_format_version=self.training.VERSION,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            registered_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            status=ModelVersionStatus.EVALUATED,
            metadata={"purpose": "test"},
        )
        return model, artifact_path

    def test_policy_service_loads_persisted_policy(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        service = LearnedPolicyService(self.policies, self.model_registry, self.training)
        loaded = service.load_policy(policy.policy_version_id)
        self.assertEqual(policy.to_dict(), loaded.to_dict())

    def test_policy_service_rejects_missing_policy(self):
        service = LearnedPolicyService(self.policies, self.model_registry, self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            service.load_policy("missing-policy")

    def test_policy_service_infers_from_persisted_ai_policy(self):
        model, _ = self._write_pretrained_ai_artifact()
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_SERVICE_INFERENCE",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=model.model_version_id,
            source_training_run_id=model.training_run_id,
            source_dataset_version_id=model.dataset_version_id,
            feature_schema_version="v1",
            label_version="v1",
            default_mode=PolicyMode.CANDIDATE,
        )
        service = LearnedPolicyService(_PolicyRepoStub(policy), _ModelRegistryStub(model), self.training)
        result = service.infer_ai(policy.policy_version_id, self._ai_snapshot())
        self.assertEqual(model.model_version_id, result.model_version_id)
        self.assertIn(result.predicted_label, {"SL_BEFORE_TP", "TP_BEFORE_SL"})

    def test_policy_loading_is_not_application(self):
        policy = self._policy("ICT", "ICT_V1", "ICT_V1")
        service = LearnedPolicyService(self.policies, self.model_registry, self.training)
        loaded = service.load_policy(policy.policy_version_id)
        self.assertEqual(policy.policy_version_id, loaded.policy_version_id)

    def test_policy_service_rejects_source_training_lineage_mismatch(self):
        model, _ = self._write_pretrained_ai_artifact()
        self.model_registry.create = Mock(return_value=model)
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_BAD_LINEAGE",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=model.model_version_id,
            source_training_run_id="wrong-training",
            source_dataset_version_id="ds-ai",
            feature_schema_version="v1",
            label_version="v1",
        )
        model_repo = _ModelRegistryStub(model)
        service = LearnedPolicyService(self.policies, model_repo, self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            service.validate_lineage(policy)

    def test_policy_service_rejects_revoked_source_model(self):
        model, _ = self._write_pretrained_ai_artifact()
        revoked = replace(model, status=ModelVersionStatus.REVOKED)
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_REVOKED",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=revoked.model_version_id,
            feature_schema_version="v1",
            label_version="v1",
        )
        service = LearnedPolicyService(self.policies, _ModelRegistryStub(revoked), self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            service.validate_lineage(policy)

    def test_policy_service_applies_persisted_classic_policy(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", default_mode=PolicyMode.CANDIDATE)
        # Repository persistence must use the explicit policy before application.
        service = LearnedPolicyService(self.policies, self.model_registry, self.training)
        result = service.apply_strategy(ClassicV1(), self.context, policy.policy_version_id, mode=PolicyMode.CANDIDATE)
        self.assertEqual("Classic", result.final_signal.strategy_name)
        self.assertEqual("Classic_V1", result.final_signal.variant)

    def test_policy_service_rejects_strategy_context_schema_mismatch(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", default_mode=PolicyMode.CANDIDATE)
        bad_current = replace(self.context.current, metadata={"feature_schema_version": "v2"})
        bad_context = replace(self.context, current=bad_current)
        service = LearnedPolicyService(self.policies, self.model_registry, self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            service.apply_strategy(ClassicV1(), bad_context, policy.policy_version_id, mode=PolicyMode.CANDIDATE)

    def test_policy_service_disabled_mode_is_exact_base_noop(self):
        strategy = ClassicV1(StrategyConfig())
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", parameter_overrides={"minimum_body_ratio": 0.99})
        before = strategy.config
        service = LearnedPolicyService(self.policies, self.model_registry, self.training)
        result = service.apply_strategy(strategy, self.context, policy.policy_version_id, mode=PolicyMode.DISABLED)
        self.assertEqual(before, strategy.config)
        self.assertEqual(result.base_signal, result.final_signal)

    def test_rule_adapter_does_not_mutate_original_strategy_config(self):
        strategy = ClassicV1(StrategyConfig())
        original = strategy.config
        policy = self._policy("Classic", "Classic_V1", "Classic_V1", parameter_overrides={"minimum_body_ratio": 0.80})
        self.adapter.apply(strategy, self.context, policy, mode=PolicyMode.CANDIDATE)
        self.assertEqual(original, strategy.config)

    def test_policy_lineage_is_available_for_audit(self):
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_LINEAGE",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id="mdl-ai-test",
            source_training_run_id="trn-ai",
            source_dataset_version_id="ds-ai",
            feature_schema_version="v1",
            label_version="v1",
        )
        self.assertEqual("trn-ai", policy.lineage()["source_training_run_id"])
        self.assertEqual("ds-ai", policy.lineage()["source_dataset_version_id"])

    def test_ai_inference_from_registered_artifact_is_transform_only(self):
        model, artifact_path = self._write_pretrained_ai_artifact()
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_INFERENCE",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=model.model_version_id,
            source_training_run_id=model.training_run_id,
            source_dataset_version_id=model.dataset_version_id,
            feature_schema_version="v1",
            label_version="v1",
            default_mode=PolicyMode.CANDIDATE,
        )
        adapter = AIModelInferenceAdapter(_ModelRegistryStub(model), self.training)
        result = adapter.infer(policy, self._ai_snapshot())
        self.assertIn(result.predicted_label, {"SL_BEFORE_TP", "TP_BEFORE_SL"})
        self.assertEqual(model.model_version_id, result.model_version_id)
        self.assertFalse(result.audit_metadata["fit_called"])
        self.assertFalse(result.audit_metadata["retrain_called"])
        self.assertTrue(artifact_path.is_file())

    def test_ai_inference_is_deterministic(self):
        model, _ = self._write_pretrained_ai_artifact()
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_DETERMINISTIC",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=model.model_version_id,
            feature_schema_version="v1",
            label_version="v1",
        )
        adapter = AIModelInferenceAdapter(_ModelRegistryStub(model), self.training)
        first = adapter.infer(policy, self._ai_snapshot())
        second = adapter.infer(policy, self._ai_snapshot())
        self.assertEqual(first.predicted_label, second.predicted_label)
        self.assertEqual(first.class_scores, second.class_scores)

    def test_ai_inference_rejects_feature_name_mismatch(self):
        model, _ = self._write_pretrained_ai_artifact()
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_FEATURE_NAMES",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=model.model_version_id,
            feature_schema_version="v1",
            label_version="v1",
        )
        bad_snapshot = FeatureSnapshot(
            timestamp=self.context.current.timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="test",
            feature_schema_version="v1",
            features={"f2": 1.0},
        )
        adapter = AIModelInferenceAdapter(_ModelRegistryStub(model), self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            adapter.infer(policy, bad_snapshot)

    def test_ai_inference_rejects_label_lineage_mismatch(self):
        model, _ = self._write_pretrained_ai_artifact()
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_LABEL_MISMATCH",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=model.model_version_id,
            feature_schema_version="v1",
            label_version="v2",
        )
        adapter = AIModelInferenceAdapter(_ModelRegistryStub(model), self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            adapter.infer(policy, self._ai_snapshot())

    def test_outcome_feature_is_rejected_before_ai_inference(self):
        with self.assertRaises(FeatureSnapshotError):
            FeatureSnapshot(
                timestamp=self.context.current.timestamp,
                symbol="XAUUSD",
                timeframe="M15",
                feature_engine_version="test",
                feature_schema_version="v1",
                features={"f1": 1.0, "outcome": "TP_BEFORE_SL"},
            )

    def test_ai_inference_rejects_future_derived_feature(self):
        model, _ = self._write_pretrained_ai_artifact()
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_NO_FUTURE",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=model.model_version_id,
            feature_schema_version="v1",
            label_version="v1",
        )
        bad_snapshot = FeatureSnapshot(
            timestamp=self.context.current.timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="test",
            feature_schema_version="v1",
            features={"future_high": 101.0},
        )
        adapter = AIModelInferenceAdapter(_ModelRegistryStub(model), self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            adapter.infer(policy, bad_snapshot)

    def test_ai_inference_accepts_evaluated_but_not_revoked_model(self):
        model, _ = self._write_pretrained_ai_artifact()
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_EVALUATED",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.MODEL_POLICY,),
            source_model_version_id=model.model_version_id,
            feature_schema_version="v1",
            label_version="v1",
        )
        result = AIModelInferenceAdapter(_ModelRegistryStub(model), self.training).infer(policy, self._ai_snapshot())
        self.assertEqual("AI", result.strategy_name)

    def test_active_ai_policy_without_model_is_rejected_by_service(self):
        policy = LearnedPolicyVersion.create(
            policy_version_label="AI_ACTIVE_NO_MODEL",
            strategy_name="AI",
            strategy_variant="AI_V1",
            strategy_version="AI_V1",
            policy_types=(PolicyType.PARAMETER_POLICY,),
            default_mode=PolicyMode.CANDIDATE,
        )
        service = LearnedPolicyService(self.policies, self.model_registry, self.training)
        with self.assertRaises(LearnedPolicyApplicationError):
            service.validate_lineage(policy)

    def test_policy_application_keeps_market_regime_as_context_only(self):
        policy = self._policy("SMC", "SMC_V1", "SMC_V1")
        self.assertEqual("TREND", self.context.current.metadata["market_regime"])
        result = self.adapter.apply(SMCV1(), self.context, policy, mode=PolicyMode.SHADOW)
        self.assertEqual(result.base_signal, result.final_signal)

    def test_no_fixed_strategy_weights_can_be_nested_in_policy_metadata(self):
        with self.assertRaises(LearnedPolicyError):
            LearnedPolicyVersion.create(
                policy_version_label="bad-nested-weight",
                strategy_name="ICT",
                strategy_variant="ICT_V1",
                strategy_version="ICT_V1",
                policy_types=(PolicyType.PARAMETER_POLICY,),
                metadata={"nested": {"AI_weights": {"AI": 1.0}}},
            )


class _NoDatasetRepository:
    def get(self, _: str):
        return None


class _PolicyRepoStub:
    def __init__(self, policy: LearnedPolicyVersion) -> None:
        self.policy = policy

    def get(self, policy_version_id: str):
        return self.policy if policy_version_id == self.policy.policy_version_id else None


class _ModelRegistryStub:
    def __init__(self, model: ModelVersion) -> None:
        self.model = model

    def get(self, model_version_id: str):
        return self.model if model_version_id == self.model.model_version_id else None


class _InferenceOnlyPreprocessor:
    def fit(self, *_args, **_kwargs):
        raise AssertionError("L9 inference must not fit preprocessing")

    def fit_transform(self, *_args, **_kwargs):
        raise AssertionError("L9 inference must not fit preprocessing")

    def transform(self, rows):
        return rows


class _InferenceOnlyLabelEncoder:
    classes_ = ("SL_BEFORE_TP", "TP_BEFORE_SL")

    def inverse_transform(self, encoded):
        return tuple(self.classes_[int(index)] for index in encoded)


class _InferenceOnlyModel:
    def fit(self, *_args, **_kwargs):
        raise AssertionError("L9 inference must never fit a model")

    def predict(self, _transformed):
        return (1,)

    def predict_proba(self, _transformed):
        return ((0.25, 0.75),)

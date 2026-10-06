from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.ml_edge.labels import LabelConfig
from app.ml_edge.model import SymbolModel, make_regressor
from app.ml_edge.walkforward import DEV_END, FOLDS, OOS_END, study_dates, ts

ROOT = Path(__file__).resolve().parents[3]


def load_retrain():
    spec = importlib.util.spec_from_file_location("retrain_edge_ml", ROOT / "scripts" / "retrain_edge_ml.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["retrain_edge_ml"] = module
    spec.loader.exec_module(module)
    return module


def save_model(folder: Path, symbol: str, enabled: bool) -> None:
    x = np.zeros((10, 1), dtype=np.float32)
    g = make_regressor("small").fit(np.r_[x, x + 1], np.r_[np.zeros(10), np.ones(10)])
    SymbolModel(symbol, "global", "small", LabelConfig(2.0, 48), ("a",), 0.05, enabled, 0.001, g, None).save(folder)


class RetrainPublishTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retrain = load_retrain()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.sources = {"A": root / "trained" / "A", "B1": root / "trained" / "B1"}
        save_model(self.sources["A"] / "AUDUSD", "AUDUSD", True)
        save_model(self.sources["A"] / "EURUSD", "EURUSD", False)
        save_model(self.sources["B1"] / "EURJPY", "EURJPY", True)
        self.live = root / "models" / "edge_ml"
        save_model(self.live / "B1" / "XAUUSD", "XAUUSD", True)  # published before, not enabled any more
        (self.live / "performance.json").write_text("{}", encoding="utf-8")
        self.now = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def live_keys(self) -> set[str]:
        return {f"{p.parent.name}/{p.name}" for p in self.live.glob("*/*") if p.is_dir()}

    def test_publishes_enabled_models_and_yen_copies(self) -> None:
        report = self.retrain.publish(self.sources, self.live, now=self.now)
        self.assertEqual(self.live_keys(), {"A/AUDUSD", "B1/EURJPY", "B1/AUDJPY", "B1/CADJPY"})
        self.assertEqual(report.skipped, ["A/EURUSD"])
        self.assertEqual(report.removed, ["B1/XAUUSD"])
        for symbol in ("AUDJPY", "CADJPY"):
            model = SymbolModel.load(self.live / "B1" / symbol)  # checksum still valid
            self.assertEqual(model.symbol, symbol)
            self.assertEqual(model.info["source_model"], "models/edge_ml/B1/EURJPY")
        self.assertTrue((self.live / "performance.json").exists())
        self.assertTrue((report.backup / "B1" / "XAUUSD" / "model.joblib").exists())
        self.assertFalse(self.live.with_name("edge_ml.new").exists())

    def test_dry_run_changes_nothing(self) -> None:
        report = self.retrain.publish(self.sources, self.live, dry_run=True, now=self.now)
        self.assertEqual(len(report.published), 4)
        self.assertEqual(self.live_keys(), {"B1/XAUUSD"})
        self.assertIsNone(report.backup)

    def test_broken_model_leaves_live_folder_untouched(self) -> None:
        (self.sources["A"] / "AUDUSD" / "model.joblib").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.retrain.publish(self.sources, self.live, now=self.now)
        self.assertEqual(self.live_keys(), {"B1/XAUUSD"})
        self.assertFalse(self.live.with_name("edge_ml.new").exists())

    def test_no_enabled_model_is_an_error(self) -> None:
        with self.assertRaises(RuntimeError):
            self.retrain.publish({"A": self.sources["A"] / "EURUSD_only"}, self.live, now=self.now)
        self.assertEqual(self.live_keys(), {"B1/XAUUSD"})


class StudyDateTests(unittest.TestCase):
    def test_default_dates_are_the_published_study(self) -> None:
        self.assertEqual((DEV_END, OOS_END, FOLDS), study_dates({}))
        self.assertEqual(DEV_END, ts("2023-11-14T23:02:00"))
        self.assertEqual(FOLDS[0][0], ts("2021-01-01"))

    def test_override_derives_yearly_folds(self) -> None:
        dev, oos, folds = study_dates({"EDGE_HUNTER_ML_DEV_END": "2025-08-31"})
        self.assertEqual(dev, ts("2025-08-31"))
        self.assertEqual(oos, dev + 410 * 86400)
        self.assertEqual(folds, ((ts("2023-01-01"), ts("2024-01-01")), (ts("2024-01-01"), ts("2025-01-01")), (ts("2025-01-01"), dev)))

    def test_short_last_year_is_merged(self) -> None:
        _, _, folds = study_dates({"EDGE_HUNTER_ML_DEV_END": "2025-02-15", "EDGE_HUNTER_ML_OOS_END": "2026-01-01"})
        self.assertEqual(folds[0][0], ts("2022-01-01"))
        self.assertEqual(folds[2], (ts("2024-01-01"), ts("2025-02-15")))

    def test_retrain_date_env_matches(self) -> None:
        env = load_retrain().date_env("2025-08-31")
        dev, oos, _ = study_dates(env)
        self.assertEqual(dev, ts("2025-08-31"))
        self.assertGreater(ts(env["EDGE_HUNTER_ML_NEW_FORWARD"]), oos)
        self.assertGreater(ts(env["EDGE_HUNTER_ML_DATA_END_REQUIRED"]), ts(env["EDGE_HUNTER_ML_NEW_FORWARD"]))


if __name__ == "__main__":
    unittest.main()

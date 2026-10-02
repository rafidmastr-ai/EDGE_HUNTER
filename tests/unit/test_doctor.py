from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from config.config_hunter import load_settings

ROOT = Path(__file__).resolve().parents[2]


def load_doctor():
    spec = importlib.util.spec_from_file_location("edge_hunter_doctor", ROOT / "scripts" / "edge_hunter_doctor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["edge_hunter_doctor"] = module
    spec.loader.exec_module(module)
    return module


class DoctorTests(unittest.TestCase):
    def test_reports_every_missing_piece_with_a_fix(self) -> None:
        doctor = load_doctor()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            settings = replace(load_settings(), data_mode="local", live_provider_name="none", live_provider_api_key=None,
                               edge_ml_models_dir=tmp / "models", edge_ml_store_dir=tmp / "store")
            report = doctor.run(tmp, offline=True, settings=settings, raw_dir=tmp / "raw")
        text = "\n".join(p + " " + f for p, f in report.problems)
        self.assertIn("ملفات قديمة أو ناقصة", text)
        self.assertIn("وضع البيانات = local", text)
        self.assertIn("مزوّد Twelve Data غير مفعّل", text)
        self.assertIn("النماذج المحمّلة 0 من 10", text)
        self.assertIn("تاريخ M1 غير كافٍ", text)
        self.assertTrue(all(fix for _, fix in report.problems))

    def test_the_repository_itself_passes_the_offline_file_checks(self) -> None:
        doctor = load_doctor()
        settings = replace(load_settings(), data_mode="live", edge_ml_store_dir=Path(tempfile.mkdtemp()))
        report = doctor.run(ROOT, offline=True, settings=settings)
        text = "\n".join(p for p, _ in report.problems)
        self.assertNotIn("ملفات قديمة", text)
        self.assertNotIn("النماذج المحمّلة", text)
        self.assertNotIn("تاريخ M1", text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.data.pipeline import HistoricalDataPipeline
from app.features.engine import FeatureEngine
from tests.unit.features.test_engine import make_bars


class FeatureEngineIntegrationTests(unittest.TestCase):
    def test_feature_engine_consumes_phase02_canonical_bars(self) -> None:
        pipeline = HistoricalDataPipeline()
        # The integration boundary uses canonical records produced by Phase 02.
        bars = make_bars(205)
        engine = FeatureEngine()
        series = engine.compute(bars, "XAUUSD", "M15")
        self.assertEqual(len(series.snapshots), len(bars))
        self.assertEqual(series.snapshots[-1].get("price.close"), float(bars[-1].close))
        self.assertTrue(series.snapshots[-1].warmup_complete)
        self.assertIsNotNone(pipeline)


if __name__ == "__main__":
    unittest.main()

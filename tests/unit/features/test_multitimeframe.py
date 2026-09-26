from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.engine import FeatureEngine
from app.features.multitimeframe import MultiTimeframeAligner
from .test_engine import make_bars


class MultiTimeframeTests(unittest.TestCase):
    def test_backward_asof_alignment_never_uses_future_snapshot(self) -> None:
        engine = FeatureEngine()
        base = engine.compute(make_bars(20), "XAUUSD", "M1")
        higher = engine.compute(make_bars(4), "XAUUSD", "M5")
        views = MultiTimeframeAligner().align(base, {"M5": higher})
        first_aligned = views[0].get("M5")
        self.assertIsNotNone(first_aligned)
        self.assertEqual(first_aligned.timestamp, views[0].base_timestamp)  # type: ignore[union-attr]
        self.assertIsNotNone(views[6].get("M5"))
        self.assertLessEqual(views[6].get("M5").timestamp, views[6].base_timestamp)  # type: ignore[union-attr]

        future_higher = make_bars(5)
        future_higher = list(future_higher)
        # Move the last higher-timeframe record beyond the last base timestamp.
        from dataclasses import replace
        future_higher[-1] = replace(
            future_higher[-1],
            timestamp=views[-1].base_timestamp + timedelta(minutes=1),
        )
        future_series = engine.compute(future_higher, "XAUUSD", "M5")
        future_views = MultiTimeframeAligner().align(base, {"M5": future_series})
        self.assertLessEqual(future_views[-1].get("M5").timestamp, future_views[-1].base_timestamp)  # type: ignore[union-attr]

    def test_strict_mode_excludes_equal_timestamp(self) -> None:
        base = make_bars(3)
        higher = make_bars(3)
        base_series = FeatureEngine().compute(base, "XAUUSD", "M1")
        higher_series = FeatureEngine().compute(higher, "XAUUSD", "M5")
        strict = MultiTimeframeAligner(strict_before=True)
        view = strict.align(base_series, {"M5": higher_series})[1]
        self.assertEqual(view.base_timestamp, base[1].timestamp)
        aligned = view.get("M5")
        self.assertIsNotNone(aligned)
        self.assertLess(aligned.timestamp, view.base_timestamp)  # type: ignore[union-attr]

    def test_empty_source_produces_none_not_a_future_guess(self) -> None:
        base = FeatureEngine().compute(make_bars(3), "XAUUSD", "M1")
        empty = FeatureEngine().compute([], "XAUUSD", "H1")
        views = MultiTimeframeAligner().align(base, {"H1": empty})
        self.assertTrue(all(view.get("H1") is None for view in views))


if __name__ == "__main__":
    unittest.main()

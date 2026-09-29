from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.ml_edge.labels import LabelConfig
from app.ml_edge.volatility import VolModel, horizon_bounds, make_vol_regressor, overlay_actions, realised_vol_ratio
from app.research.intraday import Bars

START = 1_704_067_200 + 8 * 3600  # 2024-01-01 08:00 UTC (Monday)


def m1_from_closes(closes: np.ndarray, start: int = START) -> Bars:
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return Bars(start + 60 * np.arange(len(c)), o, np.maximum(o, c), np.minimum(o, c), c, 60)


class RealisedVolTests(unittest.TestCase):
    def test_uses_only_the_trade_window(self) -> None:
        rng = np.random.default_rng(0)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 600)))
        label = LabelConfig(2.0, 4)  # 4 decision bars = 60 minutes
        decision = np.array([START + 60 * 100])
        base = realised_vol_ratio(m1_from_closes(closes), decision, np.array([0.05]), label)[0]
        later = closes.copy()
        later[100 + 61 :] *= np.exp(np.cumsum(rng.normal(0, 5e-3, len(later) - 161)))  # chaos after the window
        earlier = closes.copy()
        earlier[:99] *= np.exp(rng.normal(0, 5e-3, 99))  # chaos before the decision
        self.assertAlmostEqual(realised_vol_ratio(m1_from_closes(later), decision, np.array([0.05]), label)[0], base)
        self.assertAlmostEqual(realised_vol_ratio(m1_from_closes(earlier), decision, np.array([0.05]), label)[0], base)

    def test_window_matches_label_horizon(self) -> None:
        m1 = m1_from_closes(np.full(2000, 100.0))
        starts, ends = horizon_bounds(m1, np.array([START + 600]), LabelConfig(2.0, 48, max_hold_minutes=720))
        self.assertEqual(int(ends[0] - starts[0]), 720)


class OverlayTests(unittest.TestCase):
    def test_thresholds_use_only_previous_days(self) -> None:
        times = START + 900 * np.arange(96 * 40)
        f = np.ones(len(times))
        f[96 * 30 :] = 5.0  # a volatility spike from day 30
        take, risk = overlay_actions(times, f, "size+filter90")
        self.assertTrue(np.all(take[: 96 * 30]))
        self.assertFalse(take[96 * 30])  # spike vs the previous 60 days -> skipped
        self.assertAlmostEqual(risk[96 * 30], 0.5)  # size clipped to 0.5 x
        self.assertTrue(np.all(risk[: 96 * 20] == 1.0))  # warm-up: base risk
        none_take, none_risk = overlay_actions(times, f, "none")
        self.assertTrue(none_take.all() and np.all(none_risk == 1.0))

    def test_save_load_roundtrip(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2000, 3)).astype(np.float32)
        model = VolModel(LabelConfig(2.0, 16), ("a", "b", "c"), make_vol_regressor().fit(x, x[:, 0]))
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            loaded = VolModel.load(Path(tmp))
            self.assertTrue(np.allclose(loaded.forecast_ratio(x), model.forecast_ratio(x)))
            (Path(tmp) / "vol_model.joblib").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                VolModel.load(Path(tmp))


if __name__ == "__main__":
    unittest.main()

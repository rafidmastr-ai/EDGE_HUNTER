"""Fewer losing trades for the B1 EURJPY model on yen pairs, chosen WITHOUT the test data.

Model: the saved B1 EURJPY model (hold to TP/SL, 12 h cap, Friday exit, swap), unchanged.
Only trade-management rules are searched, on a small pre-declared grid:
  - threshold strictness: keep 5 % (current), 3 %, 2 % of the model's own past edges
  - direction: both / SELL only / BUY only
  - break-even: none, or move the stop to entry after +0.5 R / +0.75 R
  - target: 0.8 R, 1.0 R (current), 1.25 R (stop stays 2 x ATR)
Selection set (models fixed, never trained on it): EURJPY 2023-11-14 -> 2025-08-31.
Rule: best net R / max drawdown with >= 100 trades, and it must beat the current rules
on that set. The choice is then evaluated ONCE on the test sets:
  EURJPY 2025-09 -> 2026-09, AUDJPY and CADJPY 2026-04 -> 2026-09 (pairs never seen).

    python scripts/ml_edge_improve_jpy.py
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml_edge.data import TRAINING_SYMBOLS, cost_price, load_symbol, swap_price  # noqa: E402
from app.ml_edge.features import build_all  # noqa: E402
from app.ml_edge.model import SymbolModel  # noqa: E402
from app.ml_edge.walkforward import DEV_END, execution_config, ts  # noqa: E402
from app.research.intraday import DAY, full_metrics, make_orders, simulate  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
CACHE = PROJECT_ROOT / "data" / "processed" / "edge_ml_cache"
RESULTS = PROJECT_ROOT / "docs" / "results"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "edge_ml_v2" / "B1" / "EURJPY"
NEW_PAIRS = ("AUDJPY", "CADJPY")
SELECTION_END = ts("2025-09-01")
FULL_WINDOW_BARS = 96 * 60
GRID = {"keep": (0.05, 0.03, 0.02), "side": ("both", "sell", "buy"), "breakeven": (None, 0.5, 0.75), "target_rr": (1.0, 0.8, 1.25)}
CURRENT = {"keep": 0.05, "side": "both", "breakeven": None, "target_rr": 1.0}
MIN_TRADES = 100
RANDOM_RUNS = 200


def iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def stats(trades) -> dict:
    if not len(trades):
        return {"trades": 0}
    m = full_metrics(trades)
    r = trades.r_net
    out = {k: m.get(k) for k in ("trades", "win_rate", "avg_r", "profit_factor", "net_r", "avg_r_ci95", "max_drawdown_r")}
    out["losing_trades"] = int((r < -0.05).sum())  # a break-even exit (about -cost) is not counted as a loss
    out["full_losses"] = int((r <= -0.9).sum())
    out["breakeven_exits"] = int((np.abs(r) <= 0.05).sum())
    out["loss_rate"] = round(out["losing_trades"] / len(r), 4)
    out["ret_over_dd"] = round(m["net_r"] / m["max_drawdown_r"], 2) if m.get("max_drawdown_r") else None
    return out


class Series:
    """Decision rows of one pair for the B1 EURJPY model (features built with the six training symbols)."""

    def __init__(self, pair: str, symbols: dict, frames: dict, start: int, end: int | None = None) -> None:
        f = frames[pair]
        rows = f.tradable & (f.close_time > start) & (f.close_time < (end or 2**62))
        self.pair, self.data = pair, symbols[pair]
        self.times, self.x, self.atr = f.close_time[rows], f.x[rows], f.atr[rows]
        saved = SymbolModel.load(MODEL_DIR)
        self.model = replace(saved, symbol=pair, enabled=True, cost_price=cost_price(pair), swap_per_night=swap_price(pair))
        self._cache: dict = {}

    def direction(self, keep: float) -> np.ndarray:
        if keep not in self._cache:
            self._cache[keep] = replace(self.model, keep=keep).decide(self.times, self.x, self.atr)[0]
        return self._cache[keep]

    def run(self, cfg: dict, window: tuple[int, int] = (0, 2**62), direction: np.ndarray | None = None):
        d = self.direction(cfg["keep"]).copy() if direction is None else direction.copy()
        if cfg["side"] == "sell":
            d[d > 0] = 0
        elif cfg["side"] == "buy":
            d[d < 0] = 0
        sel = (d != 0) & (self.times >= window[0]) & (self.times < window[1])
        n = int(sel.sum())
        orders = make_orders(self.times[sel], d[sel], np.full(n, np.nan), np.full(n, np.nan),
                             stop_distance=self.model.label.barrier_atr * self.atr[sel], target_rr=np.full(n, cfg["target_rr"]),
                             breakeven_r=cfg["breakeven"])
        return simulate(self.data.m1, orders, execution_config(self.pair, self.model.label))


def name(cfg: dict) -> str:
    be = "none" if cfg["breakeven"] is None else f"+{cfg['breakeven']}R"
    return f"keep {cfg['keep']:.0%} | {cfg['side']} | break-even {be} | target {cfg['target_rr']}R"


def main() -> int:
    symbols = {s: load_symbol(RAW / s, CACHE) for s in TRAINING_SYMBOLS}
    for p in NEW_PAIRS:
        symbols[p] = load_symbol(RAW / p)
    frames = build_all(symbols, session_start_hour=1)
    ej = Series("EURJPY", symbols, frames, DEV_END)  # continuous since 2023-11 (threshold history as in research)

    # ---- selection on EURJPY 2023-11 -> 2025-08 only
    grid = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    scores = []
    for cfg in grid:
        s = stats(ej.run(cfg, (DEV_END, SELECTION_END)))
        scores.append({"config": name(cfg), **cfg, **s})
    current = next(s for s in scores if all(s[k] == v for k, v in CURRENT.items()))
    eligible = [s for s in scores if s["trades"] >= MIN_TRADES and s.get("ret_over_dd") is not None]
    best = max(eligible, key=lambda s: s["ret_over_dd"])
    if best["ret_over_dd"] <= current["ret_over_dd"]:
        best = current
    chosen = {k: best[k] for k in GRID}
    print("selection current:", current)
    print("selection chosen :", best)
    fewest_losses = min((s for s in eligible if (s["avg_r"] or 0) > 0), key=lambda s: s["loss_rate"])
    print("selection lowest loss-rate with positive avg R:", fewest_losses)
    # declared second option: the best rule that still trades both directions (no bet on the yen's direction)
    both = max((s for s in eligible if s["side"] == "both"), key=lambda s: s["ret_over_dd"])
    print("selection best both-directions:", both)

    # ---- test once: EURJPY new forward, AUDJPY and CADJPY (unseen pairs)
    tests = {"EURJPY 2025-09 -> 2026-09": (ej, (SELECTION_END, 2**62))}
    for p in NEW_PAIRS:
        start = int(frames[p].close_time[FULL_WINDOW_BARS])
        tests[f"{p} 2026-04 -> 2026-09 (unseen pair)"] = (Series(p, symbols, frames, start), (0, 2**62))
    rng = np.random.default_rng(5)
    report = {"selection_set": "EURJPY 2023-11-14 -> 2025-08-31", "grid": GRID, "min_trades": MIN_TRADES,
              "current": CURRENT, "chosen": chosen, "chosen_name": name(chosen),
              "selection_scores": sorted(scores, key=lambda s: -(s.get("ret_over_dd") or -1e9)),
              "selection_lowest_loss_rate": fewest_losses, "selection_best_both_directions": both, "tests": {}}
    trade_rows = []
    for label, (series, window) in tests.items():
        item = {}
        for tag, cfg in (("current", CURRENT), ("chosen", chosen), ("lowest_loss_rate_rule", {k: fewest_losses[k] for k in GRID}),
                         ("both_directions_rule", {k: both[k] for k in GRID})):
            tr = series.run(cfg, window)
            item[tag] = stats(tr)
            if tag == "chosen":
                d = series.direction(cfg["keep"])
                rand = []
                for _ in range(RANDOM_RUNS):
                    signs = np.where(rng.random(len(d)) < 0.5, 1, -1) * (d != 0)
                    rt = series.run({**cfg, "side": "both"}, window, direction=signs)
                    rand.append(float(rt.r_net.mean()) if len(rt) else np.nan)
                item["chosen_random_direction_share_beating"] = round(float(np.mean(np.array(rand) >= (item["chosen"]["avg_r"] or 0))), 3)
                for i in range(len(tr)):
                    trade_rows.append((label, tr, i))
        report["tests"][label] = item
        print(label, json.dumps(item))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ML_EDGE_JPY_IMPROVE.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_txt(report, trade_rows, RESULTS / "ML_EDGE_JPY_IMPROVE_TRADES.txt")
    return 0


def write_txt(report: dict, rows: list, path: Path) -> None:
    L = [f"EDGE HUNTER - B1 EURJPY model on yen pairs: trade-management rule chosen on {report['selection_set']} only",
         "=" * 120,
         f"Current rule : {name(report['current'])}",
         f"Chosen rule  : {report['chosen_name']}",
         f"Both-directions rule (declared second option): {report['selection_best_both_directions']['config']}",
         f"Lowest loss-rate rule on the selection set: {report['selection_lowest_loss_rate']['config']}",
         "Entry: next minute open after the 15-minute decision bar; SL 2 x ATR14(M15); 12 h cap, Friday 20:45 UTC exit; spread 0.1 pip, swap 0.5 pip/night (Wed x3).",
         "loss = R_net < -0.05 (a break-even exit costs about the spread and is not counted as a loss).", ""]
    for label, item in report["tests"].items():
        L.append(label)
        for tag in ("current", "chosen", "both_directions_rule", "lowest_loss_rate_rule"):
            s = item[tag]
            if not s.get("trades"):
                L.append(f"  {tag:<22} no trades")
                continue
            L.append(f"  {tag:<22} trades {s['trades']:>4}  losing {s['losing_trades']:>3} ({s['loss_rate']:.0%})  full SL {s['full_losses']:>3}  "
                     f"break-even {s['breakeven_exits']:>3}  win {100 * s['win_rate']:.1f}%  avg R {s['avg_r']:+.3f}  PF {s['profit_factor']}  "
                     f"net {s['net_r']:+.1f}R  max DD {s['max_drawdown_r']}R")
        L.append(f"  random direction runs >= chosen: {100 * item['chosen_random_direction_share_beating']:.0f}%")
        L.append("")
    L.append("TRADES (chosen rule)")
    L.append(f"{'#':>4} {'set':<42} {'entry (UTC)':<16} {'dir':<4} {'entry':>9} {'SL':>9} {'TP':>9} {'exit (UTC)':<16} {'exit':>9} {'R_net':>7}")
    for k, (label, tr, i) in enumerate(rows, 1):
        d = int(tr.direction[i])
        risk = abs(tr.entry[i] - tr.stop[i])
        tp = tr.entry[i] + d * report["chosen"]["target_rr"] * risk
        L.append(f"{k:>4} {label:<42} {iso(tr.entry_time[i]):<16} {'BUY' if d > 0 else 'SELL':<4} {tr.entry[i]:>9.3f} {tr.stop[i]:>9.3f} {tp:>9.3f} "
                 f"{iso(tr.exit_time[i]):<16} {tr.exit[i]:>9.3f} {tr.r_net[i]:>+7.3f}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

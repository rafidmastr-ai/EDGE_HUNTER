"""XAUUSD intraday strategy research (read-only for the app; writes a report).

Protocol:
- PRE = all data outside the fixed test period; TEST = 2023-11-14 23:02 UTC -> 2025-01-01.
- Every configuration of every candidate is run with intraday rules (no entry after
  20:00 UTC, forced exit 20:45 UTC, one position at a time, cost per trade).
- Leave-one-year-out on PRE years estimates the whole selection procedure honestly.
- One final configuration per strategy (best PRE average R with enough trades) is
  then evaluated ONCE on TEST, at 1x and 2x cost, against a drift benchmark
  (same moments, same risk and R:R, always long / always short).

    python scripts/research_xauusd_intraday.py --cost 0.20 --out docs/results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.research.intraday import (  # noqa: E402
    ExecutionConfig,
    Trades,
    by_year,
    drift_benchmark,
    iter_grid,
    leave_one_year_out,
    m1_from_symbol_dir,
    make_orders,
    simulate,
    summarize,
)
from app.research.intraday_strategies import GENERATORS, Context  # noqa: E402

TEST_START = int(datetime(2023, 11, 14, 23, 2, tzinfo=timezone.utc).timestamp())
TEST_END = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
PRE_YEARS = [2020, 2021, 2022, 2023, 2025]
MIN_PRE_TRADES = 150


def split(trades: Trades) -> tuple[Trades, Trades]:
    test = (trades.entry_time >= TEST_START) & (trades.entry_time < TEST_END)
    return trades.select(~test), trades.select(test)


def brief(trades: Trades) -> dict:
    if not len(trades):
        return {"trades": 0}
    s = summarize(trades)
    return {"trades": s["setups"], "win_rate": s["win_rate"], "avg_r": s["avg_r"], "total_r": s["total_r"],
            "profit_factor": s["profit_factor"], "max_dd_r": s["max_drawdown_r"], "ci95": s["avg_r_ci95"]}


def v1_orders(ctx: Context) -> dict[str, object]:
    """Current Classic/SMC/ICT signals (project engine) as order streams, per strategy x timeframe."""
    from app.data.schema import CanonicalOHLC  # noqa: F401
    from app.learning.strategy_learning import HistoricalSampleBuilder, timeframe_bars
    from app.data.mt5_history import load_mt5_symbol

    _, bars, _ = load_mt5_symbol(PROJECT_ROOT / "data" / "raw" / "XAUUSD")
    streams: dict[str, list] = {}

    class Spy(HistoricalSampleBuilder):
        def _label(self, signal, symbol, timeframe, values, future):
            minutes = {"M5": 5, "M15": 15, "H1": 60}[timeframe] * 60
            close_time = int(signal.timestamp.timestamp()) // minutes * minutes + minutes
            streams.setdefault(f"V1_{signal.strategy_name}_{timeframe}", []).append(
                (close_time, 1 if signal.direction.value == "BUY" else -1, float(signal.stop_loss), float(signal.target))
            )
            return None

    for timeframe in ("M5", "M15", "H1"):
        Spy(horizon_bars=2).samples_from_bars("XAUUSD", timeframe, timeframe_bars(bars, timeframe))
    return {name: make_orders(*zip(*rows)) for name, rows in streams.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=0.20)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "results")
    parser.add_argument("--skip-v1", action="store_true")
    args = parser.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    ctx = Context(m1_from_symbol_dir(PROJECT_ROOT / "data" / "raw" / "XAUUSD"))
    print(f"loaded {len(ctx.m1)} M1 bars ({time.monotonic() - started:.0f}s)", flush=True)
    config = ExecutionConfig(cost=args.cost)
    stress = ExecutionConfig(cost=2 * args.cost)
    report: dict = {"cost": args.cost, "test_period": [TEST_START, TEST_END], "strategies": {}, "configs_tried": 0}

    families: dict[str, dict[str, tuple[dict, object]]] = {}
    for name, (generator, space) in GENERATORS.items():
        families[name] = {}
        for params in iter_grid(space):
            families[name][json.dumps(params, default=str)] = (params, generator(ctx, **params))
    if not args.skip_v1:
        for stream, orders in v1_orders(ctx).items():
            families.setdefault("V1_baseline", {})[stream] = ({"stream": stream}, orders)

    for name, configs in families.items():
        t0 = time.monotonic()
        rows, pre_trades = {}, {}
        for key, (params, orders) in configs.items():
            trades = simulate(ctx.m1, orders, config)
            pre, _ = split(trades)
            pre_trades[key] = pre
            rows[key] = {"params": params, "pre": brief(pre), "pre_by_year": by_year(pre), "orders": len(orders)}
        report["configs_tried"] += len(configs)
        loyo = leave_one_year_out(pre_trades, PRE_YEARS, min_trades=30)
        eligible = [k for k, row in rows.items() if row["pre"].get("trades", 0) >= MIN_PRE_TRADES]
        entry: dict = {"configs": len(configs), "loyo": loyo, "grid": rows}
        finals = eligible if name == "V1_baseline" else ([max(eligible, key=lambda k: rows[k]["pre"]["avg_r"])] if eligible else [])
        entry["final"] = {}
        for key in finals:
            params, orders = configs[key]
            trades = simulate(ctx.m1, orders, config)
            pre, test = split(trades)
            pre2, test2 = split(simulate(ctx.m1, orders, stress))
            bench = {}
            for label, d in (("long", 1), ("short", -1)):
                b_pre, b_test = split(simulate(ctx.m1, drift_benchmark(trades, orders, d, ctx.m1), ExecutionConfig(cost=args.cost, one_position=False)))
                bench[label] = {"pre_avg_r": brief(b_pre).get("avg_r"), "test_avg_r": brief(b_test).get("avg_r")}
            longs, shorts = trades.direction > 0, trades.direction < 0
            neighbours = [] if name == "V1_baseline" else [
                k for k in configs if k != key and sum(configs[k][0][p] != params[p] for p in params) == 1
            ]
            positive_neighbours = sum(1 for k in neighbours if rows[k]["pre"].get("avg_r", -1) > 0)
            entry["final"][key] = {
                "params": params,
                "pre": brief(pre),
                "pre_by_year": by_year(pre),
                "pre_2x_cost": brief(pre2),
                "test": brief(test),
                "test_2x_cost": brief(test2),
                "test_long": brief(split(trades.select(longs))[1]),
                "test_short": brief(split(trades.select(shorts))[1]),
                "drift_benchmark": bench,
                "neighbours_positive": f"{positive_neighbours}/{len(neighbours)}",
            }
        report["strategies"][name] = entry
        print(f"{name}: {len(configs)} configs, LOYO held-out avg R={loyo['held_out_avg_r']} "
              f"({loyo['positive_years']}/{len(PRE_YEARS)} years positive) [{time.monotonic() - t0:.0f}s]", flush=True)

    (out_dir / "XAUUSD_STRATEGY_RESEARCH.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"done in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

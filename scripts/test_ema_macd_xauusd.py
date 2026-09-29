"""EMA 200 + MACD strategy test on XAUUSD M5 / M15 / H1 (research only; the app is not changed).

Segments (by trade entry time, UTC):
    Training    2020-01-01 -> 2023-01-01   cost 0.20 USD
    Validation  2023-01-01 -> 2023-11-14 23:02   cost 0.30 USD
    OOS         2023-11-14 23:02 -> 2025-01-01   cost 0.30 USD
    Forward     2025-01-01 -> end of data   cost 0.30 USD
    Sensitivity: every segment again at 0.50 USD.

Versions: core (swing stop), atr (2 x ATR stop), hidden_div (core + hidden divergence).
Sensitivity runs on core only (swing k 2/5, flat percentile 10/30), never used for selection.

    python scripts/test_ema_macd_xauusd.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.research.ema_macd import confirmed_swings, ema_macd_orders, flat_threshold  # noqa: E402
from app.research.intraday import (  # noqa: E402
    ExecutionConfig,
    Trades,
    drift_benchmark,
    full_metrics,
    m1_from_symbol_dir,
    simulate,
)
from app.research.intraday_strategies import Context  # noqa: E402


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


SEGMENTS = {
    "training": (ts("2020-01-01"), ts("2023-01-01"), 0.20),
    "validation": (ts("2023-01-01"), ts("2023-11-14T23:02:00"), 0.30),
    "oos": (ts("2023-11-14T23:02:00"), ts("2025-01-01"), 0.30),
    "forward_2025": (ts("2025-01-01"), ts("2100-01-01"), 0.30),
}
TRAIN_END = SEGMENTS["training"][1]
SENSITIVITY_COST = 0.50
TIMEFRAMES = {"M5": 5, "M15": 15, "H1": 60}
VARIANTS = ("core", "atr", "hidden_div")
SENSITIVITY = ({"swing_k": 2, "flat_pct": 20}, {"swing_k": 5, "flat_pct": 20}, {"swing_k": 3, "flat_pct": 10}, {"swing_k": 3, "flat_pct": 30})


def in_segment(trades: Trades, name: str) -> Trades:
    start, end, _ = SEGMENTS[name]
    return trades.select((trades.entry_time >= start) & (trades.entry_time < end))


def evaluate(ctx: Context, orders, *, benchmark: bool = False) -> dict:
    by_cost = {cost: simulate(ctx.m1, orders, ExecutionConfig(cost=cost)) for cost in sorted({c for *_, c in SEGMENTS.values()} | {SENSITIVITY_COST})}
    stats: dict = {}
    simulate(ctx.m1, orders, ExecutionConfig(cost=0.30), stats)
    out: dict = {"orders": len(orders), "not_traded": stats}
    for name, (_, _, cost) in SEGMENTS.items():
        trades = in_segment(by_cost[cost], name)
        entry = {"cost": cost, **full_metrics(trades)}
        entry["long_avg_r"] = round(float(trades.r_net[trades.direction > 0].mean()), 4) if (trades.direction > 0).any() else None
        entry["short_avg_r"] = round(float(trades.r_net[trades.direction < 0].mean()), 4) if (trades.direction < 0).any() else None
        entry["at_0.50"] = full_metrics(in_segment(by_cost[SENSITIVITY_COST], name))
        tiny = np.abs(trades.entry - trades.stop) < cost
        entry["trades_with_stop_below_cost"] = int(tiny.sum())
        entry["avg_r_without_them"] = round(float(trades.r_net[~tiny].mean()), 4) if (~tiny).any() else None
        if benchmark and name != "training" and len(trades):
            bench = {}
            for label, d in (("always_long", 1), ("always_short", -1)):
                b = simulate(ctx.m1, drift_benchmark(trades, orders, d, ctx.m1), ExecutionConfig(cost=cost, one_position=False))
                bench[label] = full_metrics(b).get("avg_r")
            entry["drift_benchmark_avg_r"] = bench
        out[name] = entry
    return out


def decide(result: dict) -> dict:
    val, oos, fwd = result["validation"], result["oos"], result["forward_2025"]
    viable = val.get("trades", 0) >= 50 and (val.get("avg_r") or -1) > 0 and (val.get("profit_factor") or 0) >= 1.10
    reasons = []
    confirmed = False
    if not viable:
        reasons.append("validation: needs >= 50 trades, avg R > 0 and PF >= 1.10 at 0.30 USD")
    else:
        bench = oos.get("drift_benchmark_avg_r") or {}
        long_share = oos.get("long_trades", 0) / max(1, oos.get("trades", 1))
        drift = bench.get("always_long") if long_share >= 0.5 else bench.get("always_short")
        checks = {
            "oos avg R > 0": (oos.get("avg_r") or -1) > 0,
            "oos PF >= 1.10": (oos.get("profit_factor") or 0) >= 1.10,
            "oos positive at 0.50 USD": (oos["at_0.50"].get("avg_r") or -1) > 0,
            "oos beats drift benchmark": drift is not None and (oos.get("avg_r") or -1) > drift,
            "2025 not negative": (fwd.get("avg_r") or 0) >= 0,
        }
        confirmed = all(checks.values())
        reasons = [f"{name}: {'yes' if ok else 'NO'}" for name, ok in checks.items()]
    return {"viable_on_validation": viable, "edge_confirmed": confirmed, "checks": reasons}


def spot_check(ctx: Context, minutes: int, orders, threshold: float, count: int = 5) -> list[dict]:
    bars = ctx.bars(minutes)
    swings = confirmed_swings(bars, 3)
    trades = simulate(ctx.m1, orders, ExecutionConfig(cost=0.30))
    rng = random.Random(minutes)
    picks = sorted(rng.sample(range(len(trades)), min(count, len(trades))))
    rows = []
    for i in picks:
        t = int(np.searchsorted(bars.close_time, trades.signal_time[i]))
        d = int(trades.direction[i])
        swing_bar = int(swings["low_bar" if d > 0 else "high_bar"][t])
        risk = abs(trades.entry[i] - trades.stop[i])

        def iso(value: int) -> str:
            return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

        rows.append({
            "direction": "BUY" if d > 0 else "SELL",
            "signal_bar_close": iso(trades.signal_time[i]),
            "swing_bar_open": iso(bars.open_time[swing_bar]),
            "swing_confirmed_at": iso(bars.close_time[swing_bar + 3]),
            "entry_time": iso(trades.entry_time[i]),
            "entry": round(float(trades.entry[i]), 2),
            "sl": round(float(trades.stop[i]), 2),
            "tp": round(float(trades.entry[i] + d * 1.5 * risk), 2),
            "exit_time": iso(trades.exit_time[i]),
            "exit": round(float(trades.exit[i]), 2),
            "r_net": round(float(trades.r_net[i]), 3),
        })
    return rows


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if digits else f"{value:.0f}"
    return str(value)


def markdown(report: dict) -> str:
    lines = [
        "# XAUUSD — EMA 200 + MACD (12, 26, 9): M5 / M15 / H1",
        "",
        "بحث فقط؛ لم يُضف شيء إلى التطبيق. القواعد والتعريفات الثابتة في `app/research/ema_macd.py`، والتشغيل: `python scripts/test_ema_macd_xauusd.py`.",
        "",
        "**التنفيذ:** الإشارة عند إغلاق الشمعة t، الدخول بسعر افتتاح الشمعة التالية، SL/TP على شموع M1 مع SL أولاً عند الغموض، "
        "لا دخول بعد 20:00 UTC، إغلاق إجباري 20:45 UTC، صفقة واحدة مفتوحة في كل وقت. "
        "المال: مخاطرة ثابتة 1% من 10,000$ لكل صفقة بدون تراكم (1R = 100$).",
        "",
        "**الفترات:** Training 2020–2022 (تكلفة 0.20$) | Validation 2023-01 → 2023-11-14 (0.30$) | "
        "OOS 2023-11-14 → 2024-12-31 (0.30$) | Forward 2025-01 → 2025-08 (0.30$) | حساسية 0.50$.",
        "",
        f"**عدد التجارب:** {report['trials']} تشغيلاً (9 رئيسية + 12 حساسية على النسخة الأساسية). لم يُستخدم أي منها لاختيار نسخة \"أفضل\".",
        "",
        "## الحكم (قاعدة القرار محددة قبل النتائج)",
        "",
        "| الفريم | النسخة | Validation صالحة؟ | Edge مؤكد؟ | التفاصيل |",
        "|---|---|---|---|---|",
    ]
    for tf in TIMEFRAMES:
        for variant in VARIANTS:
            d = report["main"][tf][variant]["decision"]
            lines.append(f"| {tf} | {variant} | {'نعم' if d['viable_on_validation'] else 'لا'} | {'نعم' if d['edge_confirmed'] else 'لا'} | {'; '.join(d['checks'])} |")
    lines += ["", "## النتائج لكل فريم ونسخة", ""]
    cols = [("trades", "صفقات", 0), ("win_rate", "Win%", 3), ("profit_factor", "PF", 2), ("expectancy_r", "Expectancy R", 3),
            ("median_r", "Median R", 3), ("net_return_pct", "Net Return %", 1), ("net_profit_usd", "Net Profit $", 0),
            ("gross_profit_usd", "إجمالي الأرباح $", 0), ("gross_loss_usd", "إجمالي الخسائر $", 0), ("max_drawdown_pct", "Max DD %", 1),
            ("avg_duration_min", "متوسط المدة (د)", 0), ("longest_win_streak", "أطول ربح", 0), ("longest_loss_streak", "أطول خسارة", 0)]
    for tf in TIMEFRAMES:
        lines += [f"### {tf}  (عتبة EMA الأفقي = {report['thresholds'][tf]['20']:.5f})", ""]
        lines.append("| النسخة | الفترة | " + " | ".join(c[1] for c in cols) + " | 95% CI avg R | شراء/بيع avg R | عند 0.50$ avg R | معيار الانحياز (شراء/بيع) |")
        lines.append("|" + "---|" * (len(cols) + 6))
        for variant in VARIANTS:
            res = report["main"][tf][variant]
            for seg in SEGMENTS:
                m = res[seg]
                if not m.get("trades"):
                    lines.append(f"| {variant} | {seg} | 0 |" + " - |" * (len(cols) + 3))
                    continue
                vals = []
                for key, _, digits in cols:
                    v = m.get(key)
                    signed = key in ("expectancy_r", "median_r", "net_return_pct")
                    if digits == 0 or not isinstance(v, float):
                        vals.append(str(v))
                    else:
                        vals.append(fmt(v, digits) if signed else f"{v:.{digits}f}")
                ci = m.get("avg_r_ci95")
                bench = m.get("drift_benchmark_avg_r") or {}
                lines.append(
                    f"| {variant} | {seg} | " + " | ".join(vals)
                    + f" | {('[' + fmt(ci[0]) + ', ' + fmt(ci[1]) + ']') if ci else '-'}"
                    + f" | {fmt(m.get('long_avg_r'))} / {fmt(m.get('short_avg_r'))}"
                    + f" | {fmt(m['at_0.50'].get('avg_r'))}"
                    + f" | {fmt(bench.get('always_long'))} / {fmt(bench.get('always_short'))} |"
                )
        lines.append("")
    lines += ["## صفقات وقفها أصغر من التكلفة", "",
              "وقف الـSwing قد يكون قريباً جداً من سعر الدخول (مسموح بالقواعد)؛ هذه الصفقات تخسر أكثر من 1R من التكلفة وحدها. "
              "للتأكد أنها ليست سبب النتيجة:", "",
              "| الفريم | النسخة | الفترة | عددها | متوسط R بدونها |", "|---|---|---|---|---|"]
    for tf in TIMEFRAMES:
        for variant in VARIANTS:
            for seg in SEGMENTS:
                m = report["main"][tf][variant][seg]
                if m.get("trades_with_stop_below_cost"):
                    lines.append(f"| {tf} | {variant} | {seg} | {m['trades_with_stop_below_cost']} | {fmt(m.get('avg_r_without_them'))} |")
    lines += ["", "## حساسية النسخة الأساسية (للمتانة فقط)", "", "| الفريم | swing k | EMA flat pct | Validation avg R (n) | OOS avg R (n) | 2025 avg R (n) |", "|---|---|---|---|---|---|"]
    for tf in TIMEFRAMES:
        for row in report["sensitivity"][tf]:
            r = row["result"]
            lines.append(
                f"| {tf} | {row['swing_k']} | {row['flat_pct']} | {fmt(r['validation'].get('avg_r'))} ({r['validation'].get('trades', 0)}) | "
                f"{fmt(r['oos'].get('avg_r'))} ({r['oos'].get('trades', 0)}) | {fmt(r['forward_2025'].get('avg_r'))} ({r['forward_2025'].get('trades', 0)}) |"
            )
    lines += ["", "## صفقات عشوائية للتحقق اليدوي (النسخة الأساسية، UTC)", ""]
    for tf in TIMEFRAMES:
        lines.append(f"**{tf}**")
        lines.append("")
        lines.append("| اتجاه | إغلاق شمعة الإشارة | شمعة الـSwing | تأكيد الـSwing | الدخول | سعر الدخول | SL | TP | الخروج | سعر الخروج | R صافٍ |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for s in report["spot_check"][tf]:
            lines.append(f"| {s['direction']} | {s['signal_bar_close']} | {s['swing_bar_open']} | {s['swing_confirmed_at']} | {s['entry_time']} | "
                         f"{s['entry']} | {s['sl']} | {s['tp']} | {s['exit_time']} | {s['exit']} | {s['r_net']:+.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    started = time.monotonic()
    ctx = Context(m1_from_symbol_dir(PROJECT_ROOT / "data" / "raw" / "XAUUSD"))
    print(f"loaded {len(ctx.m1)} M1 bars ({time.monotonic() - started:.0f}s)", flush=True)
    report: dict = {"segments": {k: [v[0], v[1], v[2]] for k, v in SEGMENTS.items()}, "thresholds": {}, "main": {}, "sensitivity": {}, "spot_check": {}}
    trials = 0
    for tf, minutes in TIMEFRAMES.items():
        bars = ctx.bars(minutes)
        thresholds = {str(p): flat_threshold(bars, train_end=TRAIN_END, percentile=p) for p in (10, 20, 30)}
        report["thresholds"][tf] = thresholds
        report["main"][tf] = {}
        for variant in VARIANTS:
            orders, info = ema_macd_orders(ctx, minutes, variant=variant, threshold=thresholds["20"])
            result = evaluate(ctx, orders, benchmark=True)
            result["setup_info"] = info
            result["decision"] = decide(result)
            report["main"][tf][variant] = result
            trials += 1
            print(f"{tf} {variant}: orders={len(orders)} val={result['validation'].get('avg_r')} oos={result['oos'].get('avg_r')} "
                  f"-> viable={result['decision']['viable_on_validation']} confirmed={result['decision']['edge_confirmed']}", flush=True)
        report["sensitivity"][tf] = []
        for params in SENSITIVITY:
            orders, _ = ema_macd_orders(ctx, minutes, variant="core", swing_k=params["swing_k"], threshold=thresholds[str(params["flat_pct"])])
            report["sensitivity"][tf].append({**params, "result": evaluate(ctx, orders)})
            trials += 1
        core, _ = ema_macd_orders(ctx, minutes, variant="core", threshold=thresholds["20"])
        report["spot_check"][tf] = spot_check(ctx, minutes, core, thresholds["20"])
    report["trials"] = trials
    out = PROJECT_ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "XAUUSD_EMA200_MACD.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "XAUUSD_EMA200_MACD.md").write_text(markdown(report), encoding="utf-8")
    print(f"done in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

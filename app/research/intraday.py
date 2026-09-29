"""Fast intraday research harness on UTC M1 bars (NumPy).

Bars carry their *open* time; a decision taken on a bar is known at its close
(``open + period``) and is executed on the first M1 bar that opens at or after
that moment, so signals can never use prices they could not have seen.

Execution rules (``simulate``):
- one open position at a time per order stream (orders are processed in time order);
- market orders fill at the next M1 open; limit orders fill only when price trades
  through the limit (gaps fill at the better open) before their expiry;
- stop-first when stop and target fall inside the same bar;
- optional break-even move and ATR-style trailing stop;
- intraday only: no entry after ``last_entry`` UTC, forced exit at the open of the
  first M1 bar at or after ``session_exit`` UTC (and never across a UTC date);
- net R = (exit - entry) * direction / |entry - stop| - cost / |entry - stop|.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

DAY = 86_400


@dataclass(frozen=True)
class Bars:
    """OHLC arrays with open times in UTC epoch seconds and a fixed period."""

    open_time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    period: int

    def __len__(self) -> int:
        return len(self.open_time)

    @property
    def close_time(self) -> np.ndarray:
        return self.open_time + self.period


def m1_from_symbol_dir(directory: Path) -> Bars:
    """UTC M1 bars from an MT5 symbol folder (``SOURCE.json`` + yearly CSVs)."""
    from app.data.mt5_history import load_mt5_symbol

    _, bars, _ = load_mt5_symbol(Path(directory))
    return Bars(
        open_time=np.array([int(bar.timestamp.timestamp()) for bar in bars], dtype=np.int64),
        open=np.array([float(bar.open) for bar in bars]),
        high=np.array([float(bar.high) for bar in bars]),
        low=np.array([float(bar.low) for bar in bars]),
        close=np.array([float(bar.close) for bar in bars]),
        period=60,
    )


def resample(m1: Bars, minutes: int) -> Bars:
    """Epoch-aligned buckets (same alignment as ``app.research.resampling``)."""
    seconds = minutes * 60
    if seconds == m1.period:
        return m1
    key = m1.open_time // seconds
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    return Bars(
        open_time=key[starts] * seconds,
        open=m1.open[starts],
        high=np.maximum.reduceat(m1.high, starts),
        low=np.minimum.reduceat(m1.low, starts),
        close=m1.close[ends - 1],
        period=seconds,
    )


def slice_time(bars: Bars, start: int | None = None, end: int | None = None) -> Bars:
    lo = 0 if start is None else int(np.searchsorted(bars.open_time, start, "left"))
    hi = len(bars) if end is None else int(np.searchsorted(bars.open_time, end, "left"))
    return Bars(bars.open_time[lo:hi], bars.open[lo:hi], bars.high[lo:hi], bars.low[lo:hi], bars.close[lo:hi], bars.period)


# ---------------------------------------------------------------------------
# Indicators (causal: value at i uses bars <= i)
# ---------------------------------------------------------------------------
def ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    acc = values[0]
    for i, value in enumerate(values):
        acc = value if i == 0 else acc + alpha * (value - acc)
        out[i] = acc
    out[: span - 1] = np.nan
    return out


def atr(bars: Bars, length: int = 14) -> np.ndarray:
    prev = np.r_[bars.close[0], bars.close[:-1]]
    true_range = np.maximum(bars.high - bars.low, np.maximum(np.abs(bars.high - prev), np.abs(bars.low - prev)))
    out = np.full(len(bars), np.nan)
    if len(bars) >= length:
        out[length - 1] = true_range[:length].mean()
        for i in range(length, len(bars)):
            out[i] = out[i - 1] + (true_range[i] - out[i - 1]) / length
    return out


def rsi(values: np.ndarray, length: int = 14) -> np.ndarray:
    delta = np.diff(values, prepend=values[0])
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    out = np.full(len(values), np.nan)
    if len(values) <= length:
        return out
    avg_gain, avg_loss = gain[1 : length + 1].mean(), loss[1 : length + 1].mean()
    for i in range(length, len(values)):
        if i > length:
            avg_gain += (gain[i] - avg_gain) / length
            avg_loss += (loss[i] - avg_loss) / length
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def align_to(target: Bars, source: Bars, values: np.ndarray) -> np.ndarray:
    """Value of ``source`` indicator known at each ``target`` bar close (last closed source bar)."""
    index = np.searchsorted(source.close_time, target.close_time, "right") - 1
    out = np.full(len(target), np.nan)
    ok = index >= 0
    out[ok] = values[index[ok]]
    return out


def daily_levels(m1: Bars, target: Bars, *, asia_end_hour: int = 7) -> dict[str, np.ndarray]:
    """Per target bar: previous UTC day's high/low and today's Asia (00:00-asia_end) high/low.

    Asia levels are NaN until the Asia window has closed, so they are never used early.
    """
    day = m1.open_time // DAY
    days, first = np.unique(day, return_index=True)
    last = np.r_[first[1:], len(day)]
    day_high = np.maximum.reduceat(m1.high, first)
    day_low = np.minimum.reduceat(m1.low, first)
    in_asia = (m1.open_time % DAY) < asia_end_hour * 3600
    asia_high = np.full(len(days), np.nan)
    asia_low = np.full(len(days), np.nan)
    for k, (a, b) in enumerate(zip(first, last)):
        mask = in_asia[a:b]
        if mask.any():
            asia_high[k] = m1.high[a:b][mask].max()
            asia_low[k] = m1.low[a:b][mask].min()
    target_day = target.open_time // DAY
    pos = np.searchsorted(days, target_day)
    pos_ok = (pos < len(days)) & (days[np.clip(pos, 0, len(days) - 1)] == target_day)
    prev_ok = pos_ok & (pos > 0)
    out = {name: np.full(len(target), np.nan) for name in ("pdh", "pdl", "asia_high", "asia_low")}
    out["pdh"][prev_ok] = day_high[pos[prev_ok] - 1]
    out["pdl"][prev_ok] = day_low[pos[prev_ok] - 1]
    asia_known = pos_ok & ((target.close_time % DAY) >= asia_end_hour * 3600)
    out["asia_high"][asia_known] = asia_high[pos[asia_known]]
    out["asia_low"][asia_known] = asia_low[pos[asia_known]]
    return out


# ---------------------------------------------------------------------------
# Orders and execution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Orders:
    """Order stream. ``signal_time`` = moment the decision is known (a bar close)."""

    signal_time: np.ndarray
    direction: np.ndarray  # +1 / -1
    stop: np.ndarray
    target: np.ndarray  # NaN = no target (exit by trail / session)
    limit: np.ndarray | None = None  # NaN or None = market
    expiry_seconds: int = 0
    trail_distance: np.ndarray | None = None  # price distance of an ATR trail, NaN = none
    breakeven_r: float | None = None
    tag: str = ""
    stop_distance: np.ndarray | None = None  # if set: stop = fill price -/+ distance (overrides ``stop``)
    target_rr: np.ndarray | None = None  # if set: target = fill price +/- rr * risk (overrides ``target``)

    def __len__(self) -> int:
        return len(self.signal_time)


def make_orders(signal_time, direction, stop, target, **kwargs) -> Orders:
    return Orders(
        signal_time=np.asarray(signal_time, dtype=np.int64),
        direction=np.asarray(direction, dtype=np.int64),
        stop=np.asarray(stop, dtype=float),
        target=np.asarray(target, dtype=float),
        **kwargs,
    )


@dataclass(frozen=True)
class ExecutionConfig:
    cost: float = 0.20
    session_exit: tuple[int, int] | None = (20, 45)
    last_entry: tuple[int, int] | None = (20, 0)
    horizon_bars: int | None = None  # extra limit counted in execution bars
    one_position: bool = True
    stop_fills_at_gap: bool = True  # a gap through the stop fills at the (worse) open
    entry_price: Mapping[int, float] | None = None  # V1 semantics: signal_time -> decision close


@dataclass(frozen=True)
class Trades:
    signal_time: np.ndarray
    entry_time: np.ndarray
    exit_time: np.ndarray
    direction: np.ndarray
    entry: np.ndarray
    stop: np.ndarray
    exit: np.ndarray
    r_gross: np.ndarray
    r_net: np.ndarray
    outcome: np.ndarray  # 1 target, -1 initial stop, 2 moved (break-even/trail) stop, 0 time/session exit

    def __len__(self) -> int:
        return len(self.entry_time)

    def years(self) -> np.ndarray:
        return self.entry_time.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970

    def select(self, mask: np.ndarray) -> "Trades":
        return Trades(*(getattr(self, name)[mask] for name in Trades.__dataclass_fields__))


def _clock(seconds: int, hm: tuple[int, int]) -> int:
    return seconds // DAY * DAY + hm[0] * 3600 + hm[1] * 60


def simulate(
    exec_bars: Bars, orders: Orders, config: ExecutionConfig = ExecutionConfig(), stats: dict | None = None
) -> Trades:
    """Execute ``orders`` on ``exec_bars`` (normally M1) under ``config``.

    ``stats`` (optional) receives counts of orders that did not become trades, by reason.
    """
    t, o, h, l, c = exec_bars.open_time, exec_bars.open, exec_bars.high, exec_bars.low, exec_bars.close
    never = np.iinfo(np.int64).max
    rows: list[tuple] = []
    free_at = -1
    skipped = {"position_open": 0, "after_last_entry": 0, "no_session_left": 0, "limit_not_filled": 0, "invalid_stop_or_target": 0}
    for k in np.argsort(orders.signal_time, kind="stable"):
        signal_time = int(orders.signal_time[k])
        if config.one_position and signal_time < free_at:
            skipped["position_open"] += 1
            continue
        if config.last_entry is not None and signal_time > _clock(signal_time, config.last_entry):
            skipped["after_last_entry"] += 1
            continue
        d = int(orders.direction[k])
        stop, target = float(orders.stop[k]), float(orders.target[k])
        start = int(np.searchsorted(t, signal_time, "left"))
        session_end = _clock(signal_time, config.session_exit) if config.session_exit is not None else never
        end = int(np.searchsorted(t, session_end, "left")) if session_end != never else len(t)
        if config.horizon_bars is not None:
            end = min(end, start + config.horizon_bars)
        end = min(end, len(t))
        if start >= len(t) or end <= start:
            skipped["no_session_left"] += 1
            continue

        limit = None if orders.limit is None or np.isnan(orders.limit[k]) else float(orders.limit[k])
        if limit is None:
            fill = start
            entry = float(config.entry_price[signal_time]) if config.entry_price is not None else float(o[start])
        else:
            expiry = min(end, int(np.searchsorted(t, signal_time + orders.expiry_seconds, "left")))
            touched = l[start:expiry] <= limit if d > 0 else h[start:expiry] >= limit
            if not touched.any():
                skipped["limit_not_filled"] += 1
                continue
            fill = start + int(np.argmax(touched))
            entry = min(limit, float(o[fill])) if d > 0 else max(limit, float(o[fill]))
        if orders.stop_distance is not None and not np.isnan(orders.stop_distance[k]):
            stop = entry - d * float(orders.stop_distance[k])
        risk = (entry - stop) * d
        if orders.target_rr is not None and not np.isnan(orders.target_rr[k]) and risk > 1e-9 * max(1.0, abs(entry)):
            target = entry + d * float(orders.target_rr[k]) * risk
        # risk below ~1e-9 of price is floating-point noise (stop == entry), not a real stop
        if not risk > 1e-9 * max(1.0, abs(entry)) or (not np.isnan(target) and (target - entry) * d <= 0):
            skipped["invalid_stop_or_target"] += 1
            continue

        def stop_price(level: float, i: int) -> float:
            if not config.stop_fills_at_gap or i == fill:
                return level
            return min(level, float(o[i])) if d > 0 else max(level, float(o[i]))

        trail = None if orders.trail_distance is None or np.isnan(orders.trail_distance[k]) else float(orders.trail_distance[k])
        exit_index = exit_price = None
        outcome = 0
        if trail is None and orders.breakeven_r is None:
            seg_h, seg_l = h[fill:end], l[fill:end]
            stop_hit = seg_l <= stop if d > 0 else seg_h >= stop
            if np.isnan(target):
                target_hit = np.zeros(len(seg_h), bool)
            else:
                target_hit = seg_h >= target if d > 0 else seg_l <= target
                if limit is not None:
                    target_hit[0] = False  # intrabar order after a limit fill is unknown
            first_stop = int(np.argmax(stop_hit)) if stop_hit.any() else None
            first_target = int(np.argmax(target_hit)) if target_hit.any() else None
            if first_stop is not None and (first_target is None or first_stop <= first_target):
                exit_index, outcome = fill + first_stop, -1
                exit_price = stop_price(stop, exit_index)
            elif first_target is not None:
                exit_index, exit_price, outcome = fill + first_target, target, 1
        else:
            current, best = stop, entry
            for i in range(fill, end):
                if (l[i] <= current) if d > 0 else (h[i] >= current):
                    exit_index, exit_price = i, stop_price(current, i)
                    outcome = -1 if current == stop else 2
                    break
                if not np.isnan(target) and not (limit is not None and i == fill) and ((h[i] >= target) if d > 0 else (l[i] <= target)):
                    exit_index, exit_price, outcome = i, target, 1
                    break
                best = max(best, float(h[i])) if d > 0 else min(best, float(l[i]))
                if orders.breakeven_r is not None and (best - entry) * d >= orders.breakeven_r * risk:
                    current = max(current, entry) if d > 0 else min(current, entry)
                if trail is not None:
                    current = max(current, best - trail) if d > 0 else min(current, best + trail)
        if exit_index is None:
            if end < len(t) and session_end != never and t[end] < session_end + 3600 and (config.horizon_bars is None or end < start + config.horizon_bars):
                exit_index, exit_price = end, float(o[end])  # session exit at the next open
            else:
                exit_index, exit_price = end - 1, float(c[end - 1])  # market closed / horizon: last close
        gross = (exit_price - entry) * d / risk
        rows.append((signal_time, int(t[fill]), int(t[exit_index]), d, entry, stop, exit_price, gross, gross - config.cost / risk, outcome))
        free_at = int(t[exit_index]) + exec_bars.period
    if stats is not None:
        stats.update(skipped)
    names = list(Trades.__dataclass_fields__)
    ints = {"signal_time", "entry_time", "exit_time", "direction", "outcome"}
    if not rows:
        return Trades(*(np.array([], dtype=np.int64 if n in ints else float) for n in names))
    return Trades(*(np.array(col, dtype=np.int64 if n in ints else float) for n, col in zip(names, zip(*rows))))


# ---------------------------------------------------------------------------
# Metrics and year-held-out selection
# ---------------------------------------------------------------------------
def summarize(trades: Trades) -> dict:
    from app.learning.strategy_learning import StrategySample, performance

    items = [
        StrategySample("x", "x", "x", np.datetime64(int(t), "s").astype(object), {}, int(o == 1), float(r), "R")
        for t, r, o in zip(trades.entry_time, trades.r_net, trades.outcome)
    ]
    return performance(items)


def full_metrics(trades: Trades, *, capital: float = 10_000.0, risk_pct: float = 1.0) -> dict:
    """Trade statistics in R plus money figures for a fixed ``risk_pct`` of ``capital`` per trade (no compounding)."""
    n = len(trades)
    if not n:
        return {"trades": 0}
    order = np.argsort(trades.entry_time, kind="stable")
    r = trades.r_net[order]
    wins, losses = r[r > 0], r[r < 0]
    equity = np.concatenate([[0.0], np.cumsum(r)])
    drawdown_r = float(np.max(np.maximum.accumulate(equity) - equity))

    def longest(mask: np.ndarray) -> int:
        best = run = 0
        for flag in mask:
            run = run + 1 if flag else 0
            best = max(best, run)
        return best

    risk_money = capital * risk_pct / 100.0
    durations = (trades.exit_time - trades.entry_time)[order] / 60.0
    ci = summarize(trades).get("avg_r_ci95")
    return {
        "trades": n,
        "win_rate": round(float((r > 0).mean()), 4),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if len(losses) else None,
        "expectancy_r": round(float(r.mean()), 4),
        "avg_r": round(float(r.mean()), 4),
        "median_r": round(float(np.median(r)), 4),
        "avg_r_ci95": ci,
        "net_r": round(float(r.sum()), 2),
        "net_return_pct": round(float(r.sum()) * risk_pct, 2),
        "net_profit_usd": round(float(r.sum()) * risk_money, 2),
        "gross_profit_usd": round(float(wins.sum()) * risk_money, 2),
        "gross_loss_usd": round(float(losses.sum()) * risk_money, 2),
        "max_drawdown_r": round(drawdown_r, 2),
        "max_drawdown_pct": round(drawdown_r * risk_pct, 2),
        "avg_duration_min": round(float(durations.mean()), 1),
        "longest_win_streak": longest(r > 0),
        "longest_loss_streak": longest(r <= 0),
        "long_trades": int((trades.direction > 0).sum()),
        "short_trades": int((trades.direction < 0).sum()),
    }


def by_year(trades: Trades) -> dict[int, dict]:
    years = trades.years()
    out = {}
    for year in sorted(set(years.tolist())):
        chosen = trades.r_net[years == year]
        out[year] = {"trades": int(len(chosen)), "avg_r": round(float(chosen.mean()), 4), "total_r": round(float(chosen.sum()), 2)}
    return out


def leave_one_year_out(results: Mapping[str, Trades], years: Sequence[int], *, min_trades: int = 30) -> dict:
    """For each held-out year pick the config with the best avg net R on the *other* years.

    Returns the held-out trades of the picked configs (an honest out-of-sample
    estimate of the whole selection procedure) and how often each config was picked.
    """
    picked: dict[int, str] = {}
    held_out: list[np.ndarray] = []
    for year in years:
        best_name, best_score = None, -np.inf
        for name, trades in results.items():
            y = trades.years()
            mask = np.isin(y, [other for other in years if other != year])
            if mask.sum() < min_trades:
                continue
            score = float(trades.r_net[mask].mean())
            if score > best_score:
                best_name, best_score = name, score
        if best_name is None:
            continue
        picked[year] = best_name
        trades = results[best_name]
        held_out.append(trades.r_net[trades.years() == year])
    per_year = {year: round(float(r.mean()), 4) if len(r) else None for year, r in zip(picked, held_out)}
    all_r = np.concatenate(held_out) if held_out else np.array([])
    return {
        "picked": picked,
        "held_out_avg_r_by_year": per_year,
        "held_out_trades": int(len(all_r)),
        "held_out_avg_r": round(float(all_r.mean()), 4) if len(all_r) else None,
        "positive_years": int(sum(1 for value in per_year.values() if value is not None and value > 0)),
    }


def drift_benchmark(trades: Trades, orders: Orders, direction: int, exec_bars: Bars) -> Orders:
    """Market orders at the same fill moments with the same risk and R:R but a fixed ``direction``.

    Each benchmark order fills at the open of the bar where the strategy's trade was
    filled, with the stop and target placed around *that* price, so risk is identical.
    Comparing a strategy with this isolates its timing/direction skill from the
    instrument's drift (e.g. gold rising through the sample).
    """
    target_by_time = dict(zip(orders.signal_time.tolist(), orders.target.tolist()))
    rr_by_time = dict(zip(orders.signal_time.tolist(), orders.target_rr.tolist())) if orders.target_rr is not None else {}
    signal, stop, target = [], [], []
    for time_, fill_time, entry, trade_stop in zip(
        trades.signal_time.tolist(), trades.entry_time.tolist(), trades.entry.tolist(), trades.stop.tolist()
    ):
        risk = abs(entry - trade_stop)
        original_target = target_by_time.get(time_, np.nan)
        rr = rr_by_time.get(time_, np.nan)
        if np.isnan(rr) and not np.isnan(original_target):
            rr = abs(original_target - entry) / risk
        price = float(exec_bars.open[np.searchsorted(exec_bars.open_time, fill_time, "left")])
        signal.append(fill_time)
        stop.append(price - direction * risk)
        target.append(np.nan if np.isnan(rr) else price + direction * rr * risk)
    return make_orders(signal, [direction] * len(signal), stop, target)


def iter_grid(space: Mapping[str, Sequence]) -> Iterable[dict]:
    from itertools import product

    names = list(space)
    for values in product(*(space[name] for name in names)):
        yield dict(zip(names, values))


__all__ = [
    "Bars",
    "ExecutionConfig",
    "Orders",
    "Trades",
    "align_to",
    "atr",
    "by_year",
    "full_metrics",
    "daily_levels",
    "ema",
    "iter_grid",
    "leave_one_year_out",
    "m1_from_symbol_dir",
    "make_orders",
    "resample",
    "rsi",
    "simulate",
    "slice_time",
    "summarize",
]

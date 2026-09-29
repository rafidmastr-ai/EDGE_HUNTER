"""Intraday strategy candidates for the research harness (``app.research.intraday``).

Every generator takes a :class:`Context` (UTC M1 bars plus cached derived series)
and keyword parameters, and returns :class:`Orders` whose ``signal_time`` is the
close of the bar that completed the setup. Nothing here reads a bar after that
close. These are research candidates; none is wired into the live app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from functools import cached_property
from zoneinfo import ZoneInfo

import numpy as np

from app.research.intraday import DAY, Bars, Orders, align_to, atr, daily_levels, ema, make_orders, resample, rsi

NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class Context:
    m1: Bars
    _cache: dict = field(default_factory=dict, repr=False)

    def bars(self, minutes: int) -> Bars:
        key = ("bars", minutes)
        if key not in self._cache:
            self._cache[key] = resample(self.m1, minutes)
        return self._cache[key]

    def atr(self, minutes: int, length: int = 14) -> np.ndarray:
        key = ("atr", minutes, length)
        if key not in self._cache:
            self._cache[key] = atr(self.bars(minutes), length)
        return self._cache[key]

    def ema(self, minutes: int, span: int) -> np.ndarray:
        key = ("ema", minutes, span)
        if key not in self._cache:
            self._cache[key] = ema(self.bars(minutes).close, span)
        return self._cache[key]

    def levels(self, minutes: int) -> dict[str, np.ndarray]:
        key = ("levels", minutes)
        if key not in self._cache:
            self._cache[key] = daily_levels(self.m1, self.bars(minutes))
        return self._cache[key]

    def daily_atr_at(self, minutes: int) -> np.ndarray:
        """ATR(14) of UTC daily bars known at each bar close (previous completed day)."""
        key = ("datr", minutes)
        if key not in self._cache:
            daily = self.bars(1440)
            self._cache[key] = align_to(self.bars(minutes), daily, atr(daily, 14))
        return self._cache[key]

    @cached_property
    def ny_open_utc(self) -> dict[int, int]:
        """UTC epoch of 09:30 New York for every UTC day in the data."""
        out = {}
        for day in np.unique(self.m1.open_time // DAY):
            date = datetime.fromtimestamp(int(day) * DAY, tz=timezone.utc).date()
            local = datetime.combine(date, time(9, 30), tzinfo=NEW_YORK)
            out[int(day)] = int(local.timestamp())
        return out


def _seconds_of_day(times: np.ndarray) -> np.ndarray:
    return times % DAY


def _in_window(times: np.ndarray, start_hm: tuple[int, int], end_hm: tuple[int, int]) -> np.ndarray:
    sod = _seconds_of_day(times)
    return (sod >= start_hm[0] * 3600 + start_hm[1] * 60) & (sod < end_hm[0] * 3600 + end_hm[1] * 60)


def _bias(ctx: Context, bars: Bars, mode: str) -> np.ndarray:
    """+1/-1/0 per bar from a higher timeframe known at the bar close."""
    if mode == "none":
        return np.ones(len(bars), dtype=int) * 2  # 2 = both directions allowed
    if mode == "h4_ema50":
        h4 = ctx.bars(240)
        value = align_to(bars, h4, np.sign(h4.close - ctx.ema(240, 50)))
    elif mode == "h1_ema200":
        h1 = ctx.bars(60)
        value = align_to(bars, h1, np.sign(h1.close - ctx.ema(60, 200)))
    else:
        raise ValueError(f"unknown bias mode {mode!r}")
    return np.nan_to_num(value, nan=0.0).astype(int)


def _allowed(bias: np.ndarray, index: int, direction: int) -> bool:
    return bias[index] == 2 or bias[index] == direction


def _finish(rows: list[tuple], **kwargs) -> Orders:
    if not rows:
        return make_orders([], [], [], [], **kwargs)
    signal_time, direction, stop, target, *rest = zip(*rows)
    extra = {}
    if rest:
        extra["limit"] = np.array(rest[0], dtype=float)
    return make_orders(signal_time, direction, stop, target, **extra, **kwargs)


def _target(entry: float, stop: float, direction: int, rr: float | None) -> float:
    return np.nan if rr is None else entry + direction * abs(entry - stop) * rr


# ---------------------------------------------------------------------------
# 1. New York opening-range breakout
# ---------------------------------------------------------------------------
def ny_orb(ctx: Context, *, range_minutes: int = 30, stop_mode: str = "opposite", rr: float | None = 2.0,
           bias: str = "none", window_minutes: int = 150) -> Orders:
    m5 = ctx.bars(5)
    trend = _bias(ctx, m5, bias)
    day = m5.open_time // DAY
    rows = []
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(day)]
    for a, b in zip(starts, ends):
        open_utc = ctx.ny_open_utc.get(int(day[a]))
        if open_utc is None:
            continue
        t = m5.open_time[a:b]
        in_range = (t >= open_utc) & (t < open_utc + range_minutes * 60)
        if in_range.sum() < range_minutes // 5:
            continue
        hi, lo = m5.high[a:b][in_range].max(), m5.low[a:b][in_range].min()
        after = np.flatnonzero((t >= open_utc + range_minutes * 60) & (t < open_utc + (range_minutes + window_minutes) * 60))
        for j in after:
            i = a + j
            close = m5.close[i]
            direction = 1 if close > hi else -1 if close < lo else 0
            if direction == 0:
                continue
            if _allowed(trend, i, direction):
                stop = (lo if direction > 0 else hi) if stop_mode == "opposite" else (hi + lo) / 2
                rows.append((m5.close_time[i], direction, stop, _target(close, stop, direction, rr)))
            break  # first break of the day only
    return _finish(rows, tag="ny_orb")


# ---------------------------------------------------------------------------
# 2. London breakout of the Asia range
# ---------------------------------------------------------------------------
def london_asia_breakout(ctx: Context, *, max_range_atr: float = np.inf, stop_mode: str = "opposite",
                         rr: float | None = 2.0, minutes: int = 15) -> Orders:
    bars = ctx.bars(minutes)
    levels = ctx.levels(minutes)
    datr = ctx.daily_atr_at(minutes)
    window = _in_window(bars.open_time, (7, 0), (10, 0))
    day = bars.open_time // DAY
    rows, done = [], set()
    for i in np.flatnonzero(window):
        if day[i] in done:
            continue
        hi, lo = levels["asia_high"][i], levels["asia_low"][i]
        if np.isnan(hi) or np.isnan(datr[i]) or (hi - lo) > max_range_atr * datr[i]:
            continue
        close = bars.close[i]
        direction = 1 if close > hi else -1 if close < lo else 0
        if direction == 0:
            continue
        done.add(day[i])
        stop = (lo if direction > 0 else hi) if stop_mode == "opposite" else (hi + lo) / 2
        rows.append((bars.close_time[i], direction, stop, _target(close, stop, direction, rr)))
    return _finish(rows, tag="london_asia_breakout")


# ---------------------------------------------------------------------------
# 3. Liquidity sweep reversal (previous-day / Asia extremes) — "SMC V2"
# ---------------------------------------------------------------------------
def sweep_reversal(ctx: Context, *, level: str = "pd", rr: float = 2.0, entry: str = "market",
                   minutes: int = 15, buffer_atr: float = 0.1, expiry_minutes: int = 60) -> Orders:
    bars = ctx.bars(minutes)
    lv = ctx.levels(minutes)
    a = ctx.atr(minutes)
    window = _in_window(bars.open_time, (7, 0), (16, 0))
    day = bars.open_time // DAY
    names = {"pd": [("pdh", "pdl")], "asia": [("asia_high", "asia_low")], "both": [("pdh", "pdl"), ("asia_high", "asia_low")]}[level]
    rows, used = [], set()
    for i in np.flatnonzero(window):
        if np.isnan(a[i]):
            continue
        for high_name, low_name in names:
            hi_level, lo_level = lv[high_name][i], lv[low_name][i]
            if np.isnan(hi_level):
                continue
            for direction, lvl, swept in ((-1, hi_level, bars.high[i] > hi_level and bars.close[i] < hi_level),
                                          (1, lo_level, bars.low[i] < lo_level and bars.close[i] > lo_level)):
                key = (day[i], high_name, direction)
                if not swept or key in used:
                    continue
                used.add(key)
                extreme = bars.high[i] if direction < 0 else bars.low[i]
                stop = extreme - direction * buffer_atr * a[i]
                price = bars.close[i] if entry == "market" else lvl
                if (price - stop) * direction <= 0:
                    continue
                limit = np.nan if entry == "market" else lvl
                rows.append((bars.close_time[i], direction, stop, _target(price, stop, direction, rr), limit))
    return _finish(rows, expiry_seconds=expiry_minutes * 60, tag="sweep_reversal")


# ---------------------------------------------------------------------------
# 4. Volatility squeeze breakout (Bollinger inside Keltner)
# ---------------------------------------------------------------------------
def squeeze_breakout(ctx: Context, *, squeeze_bars: int = 6, stop_atr: float = 1.5, exit_mode: str = "rr2",
                     minutes: int = 15) -> Orders:
    bars = ctx.bars(minutes)
    a = ctx.atr(minutes, 20)
    close = bars.close
    mid = ema(close, 20)
    std = np.full(len(close), np.nan)
    if len(close) >= 20:
        from numpy.lib.stride_tricks import sliding_window_view

        std[19:] = sliding_window_view(close, 20).std(axis=1)
    upper_bb, lower_bb = mid + 2 * std, mid - 2 * std
    in_squeeze = (upper_bb < mid + 1.5 * a) & (lower_bb > mid - 1.5 * a)
    run = np.zeros(len(close), dtype=int)
    for i in range(1, len(close)):
        run[i] = run[i - 1] + 1 if in_squeeze[i] else 0
    window = _in_window(bars.open_time, (7, 0), (16, 0))
    rows, trail = [], []
    for i in np.flatnonzero(window[1:]) + 1:
        if run[i - 1] < squeeze_bars or np.isnan(a[i]):
            continue
        direction = 1 if close[i] > upper_bb[i] else -1 if close[i] < lower_bb[i] else 0
        if direction == 0:
            continue
        stop = close[i] - direction * stop_atr * a[i]
        rows.append((bars.close_time[i], direction, stop, _target(close[i], stop, direction, 2.0 if exit_mode == "rr2" else None)))
        trail.append(2.0 * a[i] if exit_mode == "trail" else np.nan)
    return _finish(rows, trail_distance=np.array(trail, dtype=float) if rows else None, tag="squeeze_breakout")


# ---------------------------------------------------------------------------
# 5. Trend pullback (H1 trend, M15 entry) — "Classic V2"
# ---------------------------------------------------------------------------
def trend_pullback(ctx: Context, *, stop_mode: str = "atr2", rr: float = 2.0, session: str = "07-16") -> Orders:
    m15 = ctx.bars(15)
    h1 = ctx.bars(60)
    up = (ctx.ema(60, 20) > ctx.ema(60, 50)) & (ctx.ema(60, 50) > ctx.ema(60, 200))
    down = (ctx.ema(60, 20) < ctx.ema(60, 50)) & (ctx.ema(60, 50) < ctx.ema(60, 200))
    trend = align_to(m15, h1, up.astype(float) - down.astype(float))
    e20 = ctx.ema(15, 20)
    r = rsi(m15.close, 14)
    a = ctx.atr(15)
    start, end = {"07-16": ((7, 0), (16, 0)), "00-16": ((0, 0), (16, 0))}[session]
    window = _in_window(m15.open_time, start, end)
    rows = []
    for i in np.flatnonzero(window[3:]) + 3:
        if np.isnan(trend[i]) or trend[i] == 0 or np.isnan(a[i]) or np.isnan(e20[i]) or np.isnan(r[i]):
            continue
        direction = int(trend[i])
        touched = (m15.low[i - 1] <= e20[i - 1]) if direction > 0 else (m15.high[i - 1] >= e20[i - 1])
        reclaimed = (m15.close[i] > e20[i] and r[i] > 50) if direction > 0 else (m15.close[i] < e20[i] and r[i] < 50)
        if not (touched and reclaimed):
            continue
        close = m15.close[i]
        if stop_mode == "swing3":
            stop = m15.low[i - 3 : i + 1].min() if direction > 0 else m15.high[i - 3 : i + 1].max()
        else:
            stop = close - direction * float(stop_mode[3:]) * a[i]
        if (close - stop) * direction <= 0:
            continue
        rows.append((m15.close_time[i], direction, stop, _target(close, stop, direction, rr)))
    return _finish(rows, tag="trend_pullback")


# ---------------------------------------------------------------------------
# 6. Killzone fair-value gap after an Asia sweep — "ICT V2"
# ---------------------------------------------------------------------------
def killzone_fvg(ctx: Context, *, require_sweep: bool = True, entry: str = "limit_mid", rr: float = 2.0,
                 expiry_minutes: int = 30) -> Orders:
    m5 = ctx.bars(5)
    lv = ctx.levels(5)
    zone = _in_window(m5.open_time, (7, 0), (10, 30)) | _in_window(m5.open_time, (12, 30), (16, 0))
    day = m5.open_time // DAY
    swept_low = np.zeros(len(m5), bool)
    swept_high = np.zeros(len(m5), bool)
    current_day, low_flag, high_flag = None, False, False
    for i in range(len(m5)):
        if day[i] != current_day:
            current_day, low_flag, high_flag = day[i], False, False
        if not np.isnan(lv["asia_low"][i]) and zone[i]:
            low_flag = low_flag or m5.low[i] < lv["asia_low"][i]
            high_flag = high_flag or m5.high[i] > lv["asia_high"][i]
        swept_low[i], swept_high[i] = low_flag, high_flag
    rows = []
    for i in np.flatnonzero(zone[2:]) + 2:
        bull = m5.low[i] > m5.high[i - 2] and m5.close[i] > m5.open[i]
        bear = m5.high[i] < m5.low[i - 2] and m5.close[i] < m5.open[i]
        if bull == bear:
            continue
        direction = 1 if bull else -1
        if require_sweep and not (swept_low[i] if direction > 0 else swept_high[i]):
            continue
        stop = min(m5.low[i - 2], m5.low[i - 1]) if direction > 0 else max(m5.high[i - 2], m5.high[i - 1])
        gap_mid = (m5.low[i] + m5.high[i - 2]) / 2 if direction > 0 else (m5.high[i] + m5.low[i - 2]) / 2
        price = m5.close[i] if entry == "market" else gap_mid
        if (price - stop) * direction <= 0:
            continue
        rows.append((m5.close_time[i], direction, stop, _target(price, stop, direction, rr), np.nan if entry == "market" else gap_mid))
    return _finish(rows, expiry_seconds=expiry_minutes * 60, tag="killzone_fvg")


GENERATORS = {
    "ny_orb": (ny_orb, {"range_minutes": [15, 30], "stop_mode": ["opposite", "mid"], "rr": [1.5, 2.0, None], "bias": ["none", "h4_ema50"]}),
    "london_asia_breakout": (london_asia_breakout, {"max_range_atr": [0.4, 0.7, np.inf], "stop_mode": ["opposite", "mid"], "rr": [1.5, 2.0, None]}),
    "sweep_reversal": (sweep_reversal, {"level": ["pd", "asia", "both"], "rr": [1.5, 2.0, 3.0], "entry": ["market", "retest"]}),
    "squeeze_breakout": (squeeze_breakout, {"squeeze_bars": [3, 6], "stop_atr": [1.0, 1.5, 2.0], "exit_mode": ["rr2", "trail"]}),
    "trend_pullback": (trend_pullback, {"stop_mode": ["swing3", "atr1.5", "atr2.5"], "rr": [1.5, 2.0, 3.0], "session": ["07-16", "00-16"]}),
    "killzone_fvg": (killzone_fvg, {"require_sweep": [True, False], "entry": ["market", "limit_mid"], "rr": [1.5, 2.0, 3.0]}),
}

__all__ = ["Context", "GENERATORS", "killzone_fvg", "london_asia_breakout", "ny_orb", "squeeze_breakout", "sweep_reversal", "trend_pullback"]

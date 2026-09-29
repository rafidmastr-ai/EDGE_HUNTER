"""EMA 200 + MACD(12, 26, 9) strategy for the intraday research harness.

Rules (identical on every timeframe; decision at the close of bar ``t``):

- Long: close[t] > EMA200[t]; MACD crosses above its signal line at t
  (MACD[t-1] <= Signal[t-1] and MACD[t] > Signal[t]) with MACD[t] < 0;
  EMA200 not flat. Short is the mirror (cross down with MACD[t] > 0, close below).
- Flat EMA200: |EMA200[t] / EMA200[t-20] - 1| below a threshold that is
  calibrated once per timeframe on the training period (a percentile).
- Swing (fractal, k bars): bar j is a swing low if its low is strictly below the
  lows of the k bars before and after it; it becomes usable only at the close of
  bar j + k. Swing highs are the mirror.
- Stop: ``core`` = last confirmed swing low - tick (long) / swing high + tick
  (short); ``atr`` = fill price -/+ atr_mult * ATR(14)[t]. Target = rr * risk from
  the actual fill price. Orders are market orders filled after the close of t.
- Hidden divergence filter (``hidden_div``): long only if the last two confirmed
  swing lows form a higher low in price and a lower low in MACD (short mirror).
"""

from __future__ import annotations

import numpy as np

from app.research.intraday import Bars, Orders, atr, ema, make_orders
from app.research.intraday_strategies import Context

TICK = 0.01
SLOPE_BARS = 20


def ema_raw(values: np.ndarray, span: int) -> np.ndarray:
    """Recursive EMA seeded with the first value (no masking)."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values))
    acc = values[0] if len(values) else 0.0
    for i, value in enumerate(values):
        acc = value if i == 0 else acc + alpha * (value - acc)
        out[i] = acc
    return out


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """(MACD line, signal line); NaN during the warm-up of ``slow + signal`` bars."""
    line = ema_raw(close, fast) - ema_raw(close, slow)
    sig = ema_raw(line, signal)
    warm = min(len(close), slow + signal)
    line, sig = line.copy(), sig.copy()
    line[:warm] = np.nan
    sig[:warm] = np.nan
    return line, sig


def ema_slope(ema200: np.ndarray, bars: int = SLOPE_BARS) -> np.ndarray:
    out = np.full(len(ema200), np.nan)
    if len(ema200) > bars:
        out[bars:] = ema200[bars:] / ema200[:-bars] - 1.0
    return out


def confirmed_swings(bars: Bars, k: int) -> dict[str, np.ndarray]:
    """Last confirmed swing low/high known at each bar close, plus the one before it.

    Keys: ``low``/``high`` (price), ``low_bar``/``high_bar`` (bar index j of that
    swing) and ``prev_low``/``prev_high`` + ``prev_low_bar``/``prev_high_bar``.
    A swing at j is only visible from bar j + k onwards.
    """
    n = len(bars)
    out = {name: np.full(n, np.nan) for name in ("low", "high", "prev_low", "prev_high")}
    out.update({name: np.full(n, -1, dtype=np.int64) for name in ("low_bar", "high_bar", "prev_low_bar", "prev_high_bar")})
    if n < 2 * k + 1:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    window_low = sliding_window_view(bars.low, 2 * k + 1)
    window_high = sliding_window_view(bars.high, 2 * k + 1)
    centre_low = bars.low[k : n - k]
    centre_high = bars.high[k : n - k]
    others_low = np.delete(window_low, k, axis=1).min(axis=1)
    others_high = np.delete(window_high, k, axis=1).max(axis=1)
    lows = np.flatnonzero(centre_low < others_low) + k
    highs = np.flatnonzero(centre_high > others_high) + k
    for kind, swings, prices in (("low", lows, bars.low), ("high", highs, bars.high)):
        last = prev = -1
        pointer = 0
        for t in range(n):
            while pointer < len(swings) and swings[pointer] + k <= t:
                prev, last = last, swings[pointer]
                pointer += 1
            if last >= 0:
                out[kind][t] = prices[last]
                out[f"{kind}_bar"][t] = last
            if prev >= 0:
                out[f"prev_{kind}"][t] = prices[prev]
                out[f"prev_{kind}_bar"][t] = prev
    return out


def hidden_divergence(swings: dict[str, np.ndarray], macd_line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bullish, bearish) boolean arrays from the last two confirmed swings at each bar."""
    n = len(macd_line)
    bull = np.zeros(n, bool)
    bear = np.zeros(n, bool)
    for t in range(n):
        a, b = swings["prev_low_bar"][t], swings["low_bar"][t]
        if a >= 0 and b >= 0 and not (np.isnan(macd_line[a]) or np.isnan(macd_line[b])):
            bull[t] = swings["low"][t] > swings["prev_low"][t] and macd_line[b] < macd_line[a]
        a, b = swings["prev_high_bar"][t], swings["high_bar"][t]
        if a >= 0 and b >= 0 and not (np.isnan(macd_line[a]) or np.isnan(macd_line[b])):
            bear[t] = swings["high"][t] < swings["prev_high"][t] and macd_line[b] > macd_line[a]
    return bull, bear


def flat_threshold(bars: Bars, *, train_end: int, percentile: float) -> float:
    """``percentile`` of |EMA200 slope| over bars that closed before ``train_end``."""
    slope = np.abs(ema_slope(ema(bars.close, 200)))
    mask = (bars.close_time <= train_end) & ~np.isnan(slope)
    if not mask.any():
        raise ValueError("no training bars to calibrate the flat-EMA threshold")
    return float(np.percentile(slope[mask], percentile))


def ema_macd_signals(bars: Bars, *, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(long, short) boolean arrays of the core entry conditions at each bar close."""
    close = bars.close
    e200 = ema(close, 200)
    line, sig = macd(close)
    slope = ema_slope(e200)
    prev_line, prev_sig = np.r_[np.nan, line[:-1]], np.r_[np.nan, sig[:-1]]
    with np.errstate(invalid="ignore"):
        trending = np.abs(slope) >= threshold
        cross_up = (prev_line <= prev_sig) & (line > sig) & (line < 0)
        cross_down = (prev_line >= prev_sig) & (line < sig) & (line > 0)
        long = cross_up & (close > e200) & trending
        short = cross_down & (close < e200) & trending
    return long, short


def ema_macd_orders(
    ctx: Context,
    minutes: int,
    *,
    variant: str = "core",
    threshold: float,
    swing_k: int = 3,
    atr_mult: float = 2.0,
    rr: float = 1.5,
) -> tuple[Orders, dict]:
    """Orders for ``variant`` in {"core", "atr", "hidden_div"} and a count of dropped setups."""
    if variant not in ("core", "atr", "hidden_div"):
        raise ValueError(f"unknown variant {variant!r}")
    bars = ctx.bars(minutes)
    long, short = ema_macd_signals(bars, threshold=threshold)
    swings = confirmed_swings(bars, swing_k)
    info = {"setups_long": int(long.sum()), "setups_short": int(short.sum()), "no_confirmed_swing": 0, "filtered_by_divergence": 0}
    if variant == "hidden_div":
        line, _ = macd(bars.close)
        bull, bear = hidden_divergence(swings, line)
        info["filtered_by_divergence"] = int((long & ~bull).sum() + (short & ~bear).sum())
        long, short = long & bull, short & bear
    a = atr(bars, 14)
    rows_time, rows_dir, rows_stop, rows_dist = [], [], [], []
    for t in np.flatnonzero(long | short):
        direction = 1 if long[t] else -1
        if variant == "atr":
            if np.isnan(a[t]):
                continue
            stop, distance = np.nan, atr_mult * a[t]
        else:
            level = swings["low"][t] if direction > 0 else swings["high"][t]
            if np.isnan(level):
                info["no_confirmed_swing"] += 1
                continue
            stop, distance = level - direction * TICK, np.nan
        rows_time.append(bars.close_time[t])
        rows_dir.append(direction)
        rows_stop.append(stop)
        rows_dist.append(distance)
    count = len(rows_time)
    orders = make_orders(
        rows_time, rows_dir, rows_stop, [np.nan] * count,
        stop_distance=np.array(rows_dist, dtype=float) if variant == "atr" else None,
        target_rr=np.full(count, rr),
        tag=f"ema_macd_{variant}",
    )
    return orders, info


__all__ = [
    "confirmed_swings",
    "ema_macd_orders",
    "ema_macd_signals",
    "flat_threshold",
    "hidden_divergence",
    "macd",
]

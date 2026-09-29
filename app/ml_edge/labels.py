"""Triple-barrier outcome of a long trade, evaluated on M1 bars.

For a decision at ``close_time`` (a bar close): entry = open of the first M1 bar at or
after that moment; barriers = entry +/- b * ATR; the trade ends at the first barrier
touched (stop first if both in one M1 bar), after ``h`` decision bars (h * 15 M1
bars) at that bar's close, or at the session exit (open of the first M1 bar at or
after 20:45 UTC), whichever comes first. The value is the *gross* long R in [-1, 1];
the short outcome is its negative (symmetric barriers). Costs are applied later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.research.intraday import DAY, Bars

SESSION_EXIT = 20 * 3600 + 45 * 60


@dataclass(frozen=True)
class LabelConfig:
    barrier_atr: float
    horizon_bars: int  # intraday mode: decision bars (x15 M1 bars)
    max_hold_minutes: int | None = None  # hold mode: forced exit this long after entry (no session exit)

    @property
    def hold(self) -> bool:
        return self.max_hold_minutes is not None

    @property
    def key(self) -> str:
        if self.hold:
            return f"b{self.barrier_atr:g}_hold{self.max_hold_minutes}"
        return f"b{self.barrier_atr:g}_h{self.horizon_bars}"


def triple_barrier(m1: Bars, close_time: np.ndarray, atr: np.ndarray, config: LabelConfig, decision_minutes: int = 15) -> np.ndarray:
    """Gross long R per decision (NaN when no entry is possible)."""
    return triple_barrier_exits(m1, close_time, atr, config, decision_minutes)[0]


def triple_barrier_exits(m1: Bars, close_time: np.ndarray, atr: np.ndarray, config: LabelConfig,
                         decision_minutes: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gross long R, entry time, exit time) per decision; same rules as ``intraday.simulate``.

    Intraday mode: exit by 20:45 UTC or after ``horizon_bars`` decision bars. Hold mode:
    exit at the first M1 open ``max_hold_minutes`` after entry, and never later than
    Friday 20:45 UTC (no weekend holding).
    """
    from app.research.intraday import friday_cutoff

    t, o, h, l, c = m1.open_time, m1.open, m1.high, m1.low, m1.close
    n = len(close_time)
    out = np.full(n, np.nan)
    entry_time = np.zeros(n, dtype=np.int64)
    exit_time = np.zeros(n, dtype=np.int64)
    starts = np.searchsorted(t, close_time, "left")
    for i, start in enumerate(starts.tolist()):
        if start >= len(t) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if config.hold:
            deadline = min(int(t[start]) + config.max_hold_minutes * 60, friday_cutoff(int(t[start]), (20, 45)))
            if t[start] - close_time[i] > 3600:
                continue  # market closed after the decision (weekend): no entry
            end = int(np.searchsorted(t, deadline, "left"))
        else:
            if t[start] // DAY != close_time[i] // DAY:
                continue  # next bar is another day (market closed): no entry
            deadline = int(close_time[i] // DAY * DAY + SESSION_EXIT)
            end = min(int(np.searchsorted(t, deadline, "left")), start + config.horizon_bars * decision_minutes)
        if end <= start:
            continue
        entry = o[start]
        barrier = config.barrier_atr * atr[i]
        up = h[start:end] >= entry + barrier
        down = l[start:end] <= entry - barrier
        first_up = int(np.argmax(up)) if up.any() else None
        first_down = int(np.argmax(down)) if down.any() else None
        if first_down is not None and (first_up is None or first_down <= first_up):
            out[i], exit_index = -1.0, start + first_down
        elif first_up is not None:
            out[i], exit_index = 1.0, start + first_up
        else:
            at_deadline = end < len(t) and t[end] < deadline + 3600
            if not config.hold:
                at_deadline = at_deadline and end < start + config.horizon_bars * decision_minutes
            exit_index = end if at_deadline else end - 1
            exit_price = o[end] if at_deadline else c[end - 1]
            out[i] = float(np.clip((exit_price - entry) / barrier, -1.0, 1.0))
        entry_time[i] = t[start]
        exit_time[i] = t[exit_index]
    return out, entry_time, exit_time


__all__ = ["LabelConfig", "SESSION_EXIT", "triple_barrier", "triple_barrier_exits"]

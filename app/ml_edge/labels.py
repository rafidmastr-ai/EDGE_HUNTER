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
    horizon_bars: int

    @property
    def key(self) -> str:
        return f"b{self.barrier_atr:g}_h{self.horizon_bars}"


def triple_barrier(m1: Bars, close_time: np.ndarray, atr: np.ndarray, config: LabelConfig, decision_minutes: int = 15) -> np.ndarray:
    """Gross long R per decision (NaN when no entry is possible in the session)."""
    t, o, h, l, c = m1.open_time, m1.open, m1.high, m1.low, m1.close
    out = np.full(len(close_time), np.nan)
    starts = np.searchsorted(t, close_time, "left")
    session_end = close_time // DAY * DAY + SESSION_EXIT
    ends = np.searchsorted(t, session_end, "left")
    max_len = config.horizon_bars * decision_minutes
    for i, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        if start >= len(t) or end <= start or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if t[start] // DAY != close_time[i] // DAY:
            continue  # next bar is another day (market closed): no entry
        horizon_end = min(end, start + max_len)
        entry = o[start]
        barrier = config.barrier_atr * atr[i]
        up = h[start:horizon_end] >= entry + barrier
        down = l[start:horizon_end] <= entry - barrier
        first_up = int(np.argmax(up)) if up.any() else None
        first_down = int(np.argmax(down)) if down.any() else None
        if first_down is not None and (first_up is None or first_down <= first_up):
            out[i] = -1.0
        elif first_up is not None:
            out[i] = 1.0
        else:
            exit_price = o[horizon_end] if horizon_end == end and end < len(t) and t[end] < session_end[i] + 3600 else c[horizon_end - 1]
            out[i] = float(np.clip((exit_price - entry) / barrier, -1.0, 1.0))
    return out


__all__ = ["LabelConfig", "SESSION_EXIT", "triple_barrier"]

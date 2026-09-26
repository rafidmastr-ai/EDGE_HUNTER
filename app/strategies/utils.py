"""Shared deterministic strategy utilities; no strategy-specific rules live here."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

from app.features.models import MarketAnalysisSnapshot


def adaptive_rr(strength: float, min_rr: float, max_rr: float) -> float | None:
    """Map a normalized setup-strength value into the configured R:R range.

    This is a deterministic baseline policy only. It is intentionally not an
    assertion that any R:R is optimal; later research phases may replace it.
    """
    if not (min_rr > 0 and max_rr >= min_rr and isfinite(strength)):
        return None
    normalized = min(1.0, max(0.0, strength))
    return min_rr + (max_rr - min_rr) * normalized


def normalized_strength(value: float, floor: float = 0.0, ceiling: float = 1.0) -> float:
    """Normalize a finite value to [0, 1] without introducing future data."""
    if not isfinite(value) or ceiling <= floor:
        return 0.0
    return min(1.0, max(0.0, (value - floor) / (ceiling - floor)))


def recent_low(bars: Sequence, lookback: int) -> float | None:
    if len(bars) < lookback:
        return None
    return min(float(bar.low) for bar in bars[-lookback:])


def recent_high(bars: Sequence, lookback: int) -> float | None:
    if len(bars) < lookback:
        return None
    return max(float(bar.high) for bar in bars[-lookback:])


def previous_high(bars: Sequence, offset: int = 1) -> float | None:
    if len(bars) <= offset:
        return None
    return float(bars[-1 - offset].high)


def previous_low(bars: Sequence, offset: int = 1) -> float | None:
    if len(bars) <= offset:
        return None
    return float(bars[-1 - offset].low)


def feature_number(snapshot: MarketAnalysisSnapshot, name: str) -> float | None:
    value = snapshot.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def score_flags(**flags: bool) -> tuple[float, dict[str, float]]:
    """Convert boolean confirmations into deterministic score inputs."""
    inputs = {name: 1.0 if flag else 0.0 for name, flag in flags.items()}
    return sum(inputs.values()), inputs


def valid_signal_prices(direction: str, entry: float, stop: float, target: float) -> bool:
    if direction == "BUY":
        return stop < entry < target
    if direction == "SELL":
        return target < entry < stop
    return False

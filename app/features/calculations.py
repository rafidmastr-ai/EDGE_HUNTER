"""Pure OHLC feature calculations with explicit warm-up behavior.

All functions are deterministic and only use values at indices <= the output index.
No strategy decisions live in this module.
"""

from __future__ import annotations

from math import log, sqrt
from statistics import mean
from typing import Sequence

from app.data.schema import CanonicalOHLC


OptionalFloat = float | None


def _sma(values: Sequence[float], period: int) -> list[OptionalFloat]:
    _validate_period(period)
    output: list[OptionalFloat] = [None] * len(values)
    if len(values) < period:
        return output
    window_sum = sum(values[:period])
    output[period - 1] = window_sum / period
    for index in range(period, len(values)):
        window_sum += values[index] - values[index - period]
        output[index] = window_sum / period
    return output


def ema(values: Sequence[float], period: int) -> list[OptionalFloat]:
    """Return EMA seeded by the SMA of the first full period."""
    _validate_period(period)
    output: list[OptionalFloat] = [None] * len(values)
    if len(values) < period:
        return output
    seed_index = period - 1
    current = sum(values[:period]) / period
    output[seed_index] = current
    alpha = 2.0 / (period + 1.0)
    for index in range(period, len(values)):
        current = (values[index] * alpha) + (current * (1.0 - alpha))
        output[index] = current
    return output


def rsi_wilder(closes: Sequence[float], period: int) -> list[OptionalFloat]:
    """Return Wilder RSI; the first value is available after ``period`` changes."""
    _validate_period(period)
    output: list[OptionalFloat] = [None] * len(closes)
    if len(closes) <= period:
        return output

    gains: list[float] = [0.0] * len(closes)
    losses: list[float] = [0.0] * len(closes)
    for index in range(1, len(closes)):
        change = closes[index] - closes[index - 1]
        gains[index] = max(change, 0.0)
        losses[index] = max(-change, 0.0)

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    output[period] = _rsi_from_averages(avg_gain, avg_loss)

    for index in range(period + 1, len(closes)):
        avg_gain = ((avg_gain * (period - 1)) + gains[index]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[index]) / period
        output[index] = _rsi_from_averages(avg_gain, avg_loss)
    return output


def true_range(bars: Sequence[CanonicalOHLC]) -> list[float]:
    """Calculate true range using the previous close when available."""
    ranges: list[float] = []
    previous_close: float | None = None
    for bar in bars:
        high = float(bar.high)
        low = float(bar.low)
        if previous_close is None:
            ranges.append(high - low)
        else:
            ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        previous_close = float(bar.close)
    return ranges


def atr_wilder(bars: Sequence[CanonicalOHLC], period: int) -> list[OptionalFloat]:
    """Return Wilder ATR seeded by the first ``period`` true ranges."""
    _validate_period(period)
    tr = true_range(bars)
    output: list[OptionalFloat] = [None] * len(tr)
    if len(tr) < period:
        return output
    current = sum(tr[:period]) / period
    output[period - 1] = current
    for index in range(period, len(tr)):
        current = ((current * (period - 1)) + tr[index]) / period
        output[index] = current
    return output


def macd(
    closes: Sequence[float],
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> tuple[list[OptionalFloat], list[OptionalFloat], list[OptionalFloat]]:
    """Return MACD line, signal line and histogram with explicit warm-up."""
    _validate_period(fast_period)
    _validate_period(slow_period)
    _validate_period(signal_period)
    if fast_period >= slow_period:
        raise ValueError("macd fast_period must be smaller than slow_period")

    fast = ema(closes, fast_period)
    slow = ema(closes, slow_period)
    line: list[OptionalFloat] = [None] * len(closes)
    compact: list[float] = []
    compact_indices: list[int] = []
    for index, slow_value in enumerate(slow):
        if slow_value is not None and fast[index] is not None:
            value = fast[index] - slow_value
            line[index] = value
            compact.append(value)
            compact_indices.append(index)

    compact_signal = ema(compact, signal_period)
    signal: list[OptionalFloat] = [None] * len(closes)
    histogram: list[OptionalFloat] = [None] * len(closes)
    for compact_index, source_index in enumerate(compact_indices):
        signal_value = compact_signal[compact_index]
        signal[source_index] = signal_value
        if signal_value is not None:
            histogram[source_index] = line[source_index] - signal_value  # type: ignore[operator]
    return line, signal, histogram


def rolling_std(values: Sequence[float], period: int) -> list[OptionalFloat]:
    """Return population standard deviation over the trailing window."""
    _validate_period(period)
    output: list[OptionalFloat] = [None] * len(values)
    if len(values) < period:
        return output
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        avg = mean(window)
        variance = sum((value - avg) ** 2 for value in window) / period
        output[index] = sqrt(variance)
    return output


def close_returns(closes: Sequence[float], period: int) -> list[OptionalFloat]:
    """Return simple trailing returns; requires a close from ``period`` bars back."""
    _validate_period(period)
    output: list[OptionalFloat] = [None] * len(closes)
    for index in range(period, len(closes)):
        base = closes[index - period]
        output[index] = None if base == 0 else (closes[index] / base) - 1.0
    return output


def log_returns(closes: Sequence[float]) -> list[OptionalFloat]:
    """Return one-bar log returns, using only current and previous close."""
    output: list[OptionalFloat] = [None] * len(closes)
    for index in range(1, len(closes)):
        previous = closes[index - 1]
        current = closes[index]
        if previous > 0 and current > 0:
            output[index] = log(current / previous)
    return output


def _validate_period(period: int) -> None:
    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))

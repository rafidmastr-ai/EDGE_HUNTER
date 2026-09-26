
"""Normalization of source CSV columns into the canonical OHLC contract."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Mapping

from app.data.schema import CanonicalOHLC


ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "volume": ("volume", "tick_volume", "vol"),
}


def _find(row: Mapping[str, str], aliases: tuple[str, ...]) -> str | None:
    normalized = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _decimal(value: str, field: str) -> Decimal:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, AttributeError):
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from None


def normalize_row(row: Mapping[str, str]) -> CanonicalOHLC:
    raw_timestamp = _find(row, ALIASES["timestamp"])
    if raw_timestamp is None:
        raise ValueError("Missing timestamp column")

    timestamp_text = str(raw_timestamp).strip().replace("Z", "+00:00")
    if timestamp_text.isdigit():
        # Unix epoch in milliseconds (e.g. XAUUSD exports) or seconds.
        number = int(timestamp_text)
        if number > 10_000_000_000:
            timestamp = datetime.fromtimestamp(number / 1000.0, tz=timezone.utc)
        elif number > 1_000_000_000:
            timestamp = datetime.fromtimestamp(number, tz=timezone.utc)
        else:
            raise ValueError(f"Invalid timestamp: {raw_timestamp!r}")
    else:
        try:
            timestamp = datetime.fromisoformat(timestamp_text)
        except ValueError:
            raise ValueError(f"Invalid timestamp: {raw_timestamp!r}") from None

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    values: dict[str, Decimal] = {}
    for field in ("open", "high", "low", "close"):
        value = _find(row, ALIASES[field])
        if value is None or str(value).strip() == "":
            raise ValueError(f"Missing {field} value")
        values[field] = _decimal(value, field)

    volume_raw = _find(row, ALIASES["volume"])
    volume = None if volume_raw in (None, "") else _decimal(volume_raw, "volume")

    return CanonicalOHLC(timestamp=timestamp, volume=volume, **values)

"""MT5 M1 history exported per symbol folder: ``<SYM>/<SYM>_M1_<YEAR>.csv`` + ``SOURCE.json``.

MT5 writes broker *server* time with a misleading ``+00:00`` suffix; ``SOURCE.json``
names the server time zone (``server_tz``), which is used here to convert to real UTC.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.data.cleaning import drop_synthetic_flat_runs
from app.data.loaders.csv_loader import CSVLoader
from app.data.normalizers.ohlc_normalizer import normalize_row
from app.data.schema import CanonicalOHLC

SOURCE_FILE = "SOURCE.json"


def is_mt5_symbol_dir(path: Path) -> bool:
    return path.is_dir() and (path / SOURCE_FILE).is_file()


def server_to_utc(stamp: datetime, zone: ZoneInfo) -> datetime:
    """Read the clock value as server local time; reject ambiguous or non-existent times."""
    naive = stamp.replace(tzinfo=None)
    first = naive.replace(tzinfo=zone, fold=0)
    second = naive.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        raise ValueError(f"ambiguous or non-existent server time {naive.isoformat()} in {zone.key}")
    return first.astimezone(timezone.utc)


def load_mt5_symbol(directory: Path) -> tuple[str, list[CanonicalOHLC], dict]:
    """(symbol, UTC bars sorted and cleaned, summary) for one MT5 symbol folder."""
    directory = Path(directory)
    source = json.loads((directory / SOURCE_FILE).read_text(encoding="utf-8"))
    tz_name = source.get("server_tz")
    if not tz_name:
        raise ValueError(f"{directory / SOURCE_FILE} has no server_tz")
    zone = ZoneInfo(tz_name)
    symbol = str(source.get("symbol") or directory.name).upper()
    files = sorted(directory.glob(f"{directory.name}_M1_*.csv"))
    if not files:
        raise ValueError(f"no {directory.name}_M1_*.csv files in {directory}")

    unique: dict[datetime, CanonicalOHLC] = {}
    loader = CSVLoader()
    for path in files:
        for row in loader.read_rows(path):
            bar = normalize_row(row)
            bar = replace(bar, timestamp=server_to_utc(bar.timestamp, zone))
            unique.setdefault(bar.timestamp, bar)
    ordered = [
        bar
        for bar in (unique[key] for key in sorted(unique))
        if bar.high >= max(bar.open, bar.close, bar.low) and bar.low <= min(bar.open, bar.close, bar.high) and bar.low > 0
    ]
    cleaned, removed = drop_synthetic_flat_runs(ordered)
    summary = {
        "files": [path.name for path in files],
        "server_tz": tz_name,
        "rows": len(cleaned),
        "synthetic_flat_bars_removed": removed,
        "first_utc": cleaned[0].timestamp.isoformat() if cleaned else None,
        "last_utc": cleaned[-1].timestamp.isoformat() if cleaned else None,
    }
    return symbol, cleaned, summary


__all__ = ["SOURCE_FILE", "is_mt5_symbol_dir", "load_mt5_symbol", "server_to_utc"]

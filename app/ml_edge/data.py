"""Multi-symbol M1 loading (UTC), resampling with activity columns, and trading costs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from app.research.intraday import Bars

CACHE_VERSION = 1
FX_COST_PIPS = {"base": 0.1, "sens_mid": 0.5, "sens_high": 1.5}
XAU_COST_USD = {"base": 0.10, "sens_mid": 0.30, "sens_high": 0.50}
# Estimated overnight financing per rollover (both directions), charged only when holding.
SWAP_FX_PIPS = 0.5
SWAP_XAU_USD = 0.50
# The six symbols every saved EDGE ML model was trained on (later folders in data/raw are test-only).
TRAINING_SYMBOLS = ("AUDUSD", "EURJPY", "EURUSD", "GBPUSD", "NZDUSD", "XAUUSD")


@dataclass(frozen=True)
class SymbolData:
    symbol: str
    m1: Bars
    tick_volume: np.ndarray
    spread: np.ndarray  # MT5 spread column (points), used only as a relative activity feature


def pip_size(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def cost_price(symbol: str, level: str = "base") -> float:
    """Round-trip cost in price units (owner's setting: spread only)."""
    if symbol.upper().startswith("XAU"):
        return XAU_COST_USD[level]
    return FX_COST_PIPS[level] * pip_size(symbol)


def swap_price(symbol: str, multiplier: float = 1.0) -> float:
    """Estimated swap per night in price units (owner: flat estimate; Wednesday counts 3x)."""
    if symbol.upper().startswith("XAU"):
        return SWAP_XAU_USD * multiplier
    return SWAP_FX_PIPS * pip_size(symbol) * multiplier


def discover_symbols(raw_dir: Path) -> list[str]:
    return sorted(p.parent.name for p in Path(raw_dir).glob("*/SOURCE.json"))


def _to_utc(local_epoch: np.ndarray, zone: ZoneInfo) -> np.ndarray:
    """Server wall-clock seconds -> UTC seconds; ambiguous/non-existent local hours raise."""
    hours = local_epoch // 3600
    unique = np.unique(hours)
    offsets = np.empty(len(unique), dtype=np.int64)
    for i, hour in enumerate(unique.tolist()):
        naive = datetime.fromtimestamp(hour * 3600, tz=timezone.utc).replace(tzinfo=None)
        first = naive.replace(tzinfo=zone, fold=0).utcoffset()
        second = naive.replace(tzinfo=zone, fold=1).utcoffset()
        if first != second:
            raise ValueError(f"ambiguous or non-existent server time {naive.isoformat()} in {zone.key}")
        offsets[i] = int(first.total_seconds())
    return local_epoch - offsets[np.searchsorted(unique, hours)]


def _source_key(files: list[Path]) -> str:
    digest = hashlib.sha256(str(CACHE_VERSION).encode())
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return digest.hexdigest()[:16]


DEFAULT_SERVER_TZ = "Europe/Athens"  # MetaQuotes-Demo server clock of every export in this project


def find_m1_files(raw_dir: Path, symbol: str) -> list[Path]:
    """``<raw>/<SYM>/<SYM>_M1_*.csv`` (repository layout) or ``<raw>/<SYM>_M1_*.csv`` (flat copy)."""
    raw_dir = Path(raw_dir)
    return sorted((raw_dir / symbol).glob(f"{symbol}_M1_*.csv")) or sorted(raw_dir.glob(f"{symbol}_M1_*.csv"))


def load_symbol(directory: Path, cache_dir: Path | None = None, *, files: list[Path] | None = None,
                symbol: str | None = None, default_tz: str | None = None) -> SymbolData:
    """Load MT5 M1 exports. ``SOURCE.json`` gives the server time zone; without it ``default_tz`` is used
    (``files``/``symbol`` allow the flat layout)."""
    directory = Path(directory)
    source_path = directory / "SOURCE.json"
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
    elif default_tz is not None:
        source = {"server_tz": default_tz, "symbol": symbol or directory.name}
    else:
        raise FileNotFoundError(f"missing {source_path}")
    zone = ZoneInfo(source["server_tz"])
    symbol = str(source.get("symbol") or symbol or directory.name).upper()
    files = sorted(files) if files is not None else sorted(directory.glob(f"{directory.name}_M1_*.csv"))
    if not files:
        raise ValueError(f"no M1 files in {directory}")
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{symbol}_{_source_key(files)}.npz"
        if cache.exists():
            z = np.load(cache)
            return SymbolData(symbol, Bars(z["t"], z["o"], z["h"], z["l"], z["c"], 60), z["v"], z["s"])

    stamps: list[str] = []
    values: list[tuple[float, ...]] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = [name.strip().lower() for name in next(reader)]
            idx = [header.index(name) for name in ("open", "high", "low", "close", "tick_volume", "spread")]
            ts_idx = header.index("timestamp")
            for row in reader:
                stamps.append(row[ts_idx][:19])
                values.append(tuple(float(row[i]) for i in idx))
    local = np.array(stamps, dtype="datetime64[s]").astype(np.int64)
    arr = np.array(values)
    utc = _to_utc(local, zone)
    order = np.argsort(utc, kind="stable")
    utc, arr = utc[order], arr[order]
    keep = np.r_[True, utc[1:] != utc[:-1]]
    utc, arr = utc[keep], arr[keep]
    o, h, l, c, v, s = arr.T
    valid = (h >= np.maximum.reduce([o, c, l])) & (l <= np.minimum.reduce([o, c, h])) & (l > 0)
    utc, o, h, l, c, v, s = (x[valid] for x in (utc, o, h, l, c, v, s))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, t=utc, o=o, h=h, l=l, c=c, v=v, s=s)
    return SymbolData(symbol, Bars(utc, o, h, l, c, 60), v, s)


def resample_activity(data: SymbolData, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """(sum of tick volume, mean spread) per epoch-aligned bucket, matching ``intraday.resample``."""
    key = data.m1.open_time // (minutes * 60)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    counts = np.diff(np.r_[starts, len(key)])
    return np.add.reduceat(data.tick_volume, starts), np.add.reduceat(data.spread, starts) / counts


__all__ = ["DEFAULT_SERVER_TZ", "TRAINING_SYMBOLS", "SymbolData", "find_m1_files", "cost_price", "swap_price", "discover_symbols", "load_symbol", "pip_size", "resample_activity"]

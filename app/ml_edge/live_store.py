"""Local M1 store for live EDGE ML decisions (one compressed .npz per symbol).

Seeded from the repository's MT5 history (converted to UTC) and extended with
minute bars from the live provider. Stored bars are never overwritten; live bars
have no tick volume or spread (NaN), which the live feature step neutralises.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from app.ml_edge.data import DEFAULT_SERVER_TZ, SymbolData, find_m1_files, load_symbol
from app.research.intraday import DAY, Bars

logger = logging.getLogger("edge_hunter.edge_ml")
KEEP_DAYS = 150
EMPTY_WINDOW_CODES = frozenset({"live_empty_data", "provider_400"})
RATE_LIMIT_CODES = frozenset({"provider_local_rate_limited", "provider_rate_limited", "provider_429"})
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_WAIT_SECONDS = 20.0
BACKFILL_TIMEFRAME, BACKFILL_BAR_MINUTES = "M15", 15
BACKFILL_DAYS = 95  # calendar days of M1 the models need (60-day features + 20-day threshold warm-up, with margin)
RESERVED_REQUESTS = 3  # one user analysis = 3 provider requests (M5, M15, H1)


@dataclass
class StoredSeries:
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    s: np.ndarray

    def to_symbol_data(self, symbol: str) -> SymbolData:
        return SymbolData(symbol, Bars(self.t, self.o, self.h, self.l, self.c, 60), self.v, self.s)

    @property
    def last_time(self) -> int | None:
        return int(self.t[-1]) if len(self.t) else None


def _empty() -> StoredSeries:
    f = np.array([], dtype=float)
    return StoredSeries(np.array([], dtype=np.int64), f, f, f, f, f, f)


def _recent_files(files: list[Path]) -> list[Path]:
    """Only the exports that can hold the last ``KEEP_DAYS`` (this year and the previous one):
    parsing 2020-2026 history on every first start took minutes per symbol."""
    import re

    years = {f: int(m.group(1)) for f in files if (m := re.search(r"_M1_(\d{4})", f.name))}
    if not years:
        return files
    newest = max(years.values())
    return [f for f in files if years.get(f, newest) >= newest - 1]


class M1Store:
    def __init__(self, directory: Path, symbols: tuple[str, ...], *, raw_dir: Path | None = None, keep_days: int = KEEP_DAYS) -> None:
        self.directory = Path(directory)
        self.symbols = tuple(symbols)
        self.raw_dir = raw_dir
        self.keep_days = keep_days
        self._lock = threading.Lock()
        self._series: dict[str, StoredSeries] = {}

    def _path(self, symbol: str) -> Path:
        return self.directory / f"{symbol}_M1.npz"

    def series(self, symbol: str) -> StoredSeries:
        with self._lock:
            if symbol not in self._series:
                self._series[symbol] = self._load_or_seed(symbol)
            return self._series[symbol]

    def _load_or_seed(self, symbol: str) -> StoredSeries:
        path = self._path(symbol)
        if path.exists():
            z = np.load(path)
            return StoredSeries(z["t"], z["o"], z["h"], z["l"], z["c"], z["v"], z["s"])
        files = _recent_files(find_m1_files(self.raw_dir, symbol)) if self.raw_dir is not None else []
        if files:
            folder = files[0].parent
            if not (folder / "SOURCE.json").exists():
                logger.warning("edge_ml: %s has no SOURCE.json, assuming %s server time", folder, DEFAULT_SERVER_TZ)
            try:
                data = load_symbol(folder, self.directory / "seed_cache", files=files, symbol=symbol, default_tz=DEFAULT_SERVER_TZ)
            except Exception:  # unreadable export: start empty, the backfill fetches history instead
                logger.exception("edge_ml: could not read %s history from %s", symbol, folder)
                return _empty()
            m = data.m1
            keep = m.open_time >= m.open_time[-1] - self.keep_days * DAY
            series = StoredSeries(m.open_time[keep], m.open[keep], m.high[keep], m.low[keep], m.close[keep],
                                  data.tick_volume[keep].astype(float), data.spread[keep].astype(float))
            self._save(symbol, series)
            logger.info("edge_ml store seeded %s from repository history (%d bars)", symbol, len(series.t))
            return series
        return _empty()

    def _save(self, symbol: str, series: StoredSeries) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self._path(symbol).with_suffix(".tmp.npz")
        np.savez_compressed(tmp, t=series.t, o=series.o, h=series.h, l=series.l, c=series.c, v=series.v, s=series.s)
        tmp.replace(self._path(symbol))

    def append(self, symbol: str, rows: list[tuple[int, float, float, float, float]]) -> int:
        """Add (utc_seconds, open, high, low, close) minute bars; existing timestamps are kept as they are."""
        current = self.series(symbol)
        candidates = sorted({int(r[0]): r for r in rows}.values())
        if len(current.t) and candidates:
            seen = np.isin(np.array([r[0] for r in candidates], dtype=np.int64), current.t)
            candidates = [r for r, dup in zip(candidates, seen) if not dup]
        fresh = candidates
        fresh = [r for r in fresh if r[2] >= max(r[1], r[4], r[3]) and r[3] <= min(r[1], r[4], r[2]) and r[3] > 0]
        if not fresh:
            return 0
        arr = np.array(fresh, dtype=float)
        t = np.concatenate([current.t, arr[:, 0].astype(np.int64)])
        order = np.argsort(t, kind="stable")
        nan = np.full(len(arr), np.nan)
        merged = StoredSeries(
            t[order],
            np.concatenate([current.o, arr[:, 1]])[order],
            np.concatenate([current.h, arr[:, 2]])[order],
            np.concatenate([current.l, arr[:, 3]])[order],
            np.concatenate([current.c, arr[:, 4]])[order],
            np.concatenate([current.v, nan])[order],
            np.concatenate([current.s, nan])[order],
        )
        keep = merged.t >= merged.t[-1] - self.keep_days * DAY
        merged = StoredSeries(*(getattr(merged, f)[keep] for f in ("t", "o", "h", "l", "c", "v", "s")))
        with self._lock:
            self._series[symbol] = merged
            self._save(symbol, merged)
        return len(fresh)


def _wait_for_budget(provider, pause: Callable[[float], None], reserve: int = RESERVED_REQUESTS, max_wait: float = 60.0) -> None:
    """Leave ``reserve`` requests of the provider's per-minute budget free for user analyses."""
    times, limit = getattr(provider, "_request_times", None), getattr(provider, "_rate_limit", None)
    if times is None or not isinstance(limit, int):
        return
    waited = 0.0
    while waited < max_wait:
        now = time.monotonic()
        recent = sum(1 for t in list(times) if now - t < 60.0)
        if recent <= max(0, limit - reserve - 1):
            return
        pause(5.0)
        waited += 5.0


def fetch_updates(
    store: M1Store,
    provider,
    symbol: str,
    now: datetime,
    *,
    page_minutes: int | None = None,
    max_pages: int = 30,
    pause: Callable[[float], None] = time.sleep,
    pause_seconds: float = 12.0,
) -> int:
    """Fetch closed minute bars after the last stored bar, one page (< provider max bars) at a time.

    Pauses between requests so the shared provider rate limit stays free for user analyses;
    stops quietly on provider errors (the next refresh retries).
    """
    from app.providers.models import LiveProviderError

    added = 0
    bulk = getattr(provider, "bulk_max_bars", None)
    if page_minutes is None:  # one page = the most bars the provider returns per request, minus a margin
        page_minutes = max(60, int(bulk or getattr(provider, "_max_bars", 800)) - 50)
    extra = {"max_bars": int(bulk)} if bulk else {}
    now = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    last = store.series(symbol).last_time
    start = datetime.fromtimestamp(last + 60, tz=timezone.utc) if last is not None else now - timedelta(days=5)
    for page in range(max_pages):
        if start >= now:
            break
        end = min(now - timedelta(minutes=1), start + timedelta(minutes=page_minutes))
        if end < start:
            break
        if page:
            pause(pause_seconds)
        try:
            bars = None
            for attempt in range(RATE_LIMIT_RETRIES + 1):
                try:
                    _wait_for_budget(provider, pause)
                    bars = provider.get_ohlc(symbol, "M1", start, end, **extra)
                    break
                except LiveProviderError as exc:
                    # the provider's per-minute budget is shared with user analyses: wait, do not give up
                    if exc.code not in RATE_LIMIT_CODES or attempt == RATE_LIMIT_RETRIES:
                        raise
                    pause(RATE_LIMIT_WAIT_SECONDS)
        except LiveProviderError as exc:
            # market closed in this window (weekend / holiday): Twelve Data answers
            # {"code": 400, "message": "No data is available on the specified dates"}
            if exc.code in EMPTY_WINDOW_CODES:
                start = end + timedelta(minutes=1)
                continue
            logger.warning("edge_ml fetch %s stopped: %s", symbol, exc.code)
            break
        rows = [(int(b.timestamp.astimezone(timezone.utc).timestamp()), float(b.open), float(b.high), float(b.low), float(b.close))
                for b in bars if b.timestamp.astimezone(timezone.utc) < now]
        added += store.append(symbol, rows)
        start = end + timedelta(minutes=1)
    return added


def backfill_history(
    store: M1Store,
    provider,
    symbol: str,
    now: datetime,
    *,
    target_days: float = BACKFILL_DAYS,
    max_pages: int = 40,
    pause: Callable[[float], None] = time.sleep,
    pause_seconds: float = 12.0,
) -> int:
    """Fetch older history (newest first, going back) until the store covers ``target_days``.

    Used when the repository CSV history is missing: the models need ~85 days. Old history is
    fetched as M15 bars: every model feature is built from M15/H1/H4/D1 bars and 15-minute-aligned
    daily levels, so M15 rows give the same features as M1 rows (tick volume/spread are neutral live
    anyway) for 15x fewer requests - about 2 per symbol instead of ~29. Recent bars stay M1."""
    from app.providers.models import LiveProviderError

    bulk = getattr(provider, "bulk_max_bars", None)
    page_minutes = max(60 * BACKFILL_BAR_MINUTES, (int(bulk or getattr(provider, "_max_bars", 800)) - 50) * BACKFILL_BAR_MINUTES)
    extra = {"max_bars": int(bulk)} if bulk else {}
    now = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    goal = int((now - timedelta(days=target_days)).timestamp())
    series = store.series(symbol)
    cursor = int(series.t[0]) if len(series.t) else int(now.timestamp())  # fetch everything before this
    added = 0
    for page in range(max_pages):
        if cursor <= goal:
            break
        end_s = cursor - BACKFILL_BAR_MINUTES * 60
        start_s = max(goal, end_s - page_minutes * 60)
        if page:
            pause(pause_seconds)
        start = datetime.fromtimestamp(start_s, tz=timezone.utc)
        end = datetime.fromtimestamp(end_s, tz=timezone.utc)
        try:
            bars = None
            for attempt in range(RATE_LIMIT_RETRIES + 1):
                try:
                    _wait_for_budget(provider, pause)
                    bars = provider.get_ohlc(symbol, BACKFILL_TIMEFRAME, start, end, **extra)
                    break
                except LiveProviderError as exc:
                    if exc.code not in RATE_LIMIT_CODES or attempt == RATE_LIMIT_RETRIES:
                        raise
                    pause(RATE_LIMIT_WAIT_SECONDS)
        except LiveProviderError as exc:
            if exc.code in EMPTY_WINDOW_CODES:  # weekend / holiday: step over it
                cursor = start_s
                continue
            logger.warning("edge_ml backfill %s stopped: %s", symbol, exc.code)
            break
        rows = [(int(b.timestamp.astimezone(timezone.utc).timestamp()), float(b.open), float(b.high), float(b.low), float(b.close))
                for b in bars]
        # keep only bars that close before the stored data starts (no overlap with the M1 minutes)
        rows = [r for r in rows if r[0] + BACKFILL_BAR_MINUTES * 60 <= cursor]
        added += store.append(symbol, rows)
        # a full page may be cut by the provider's count cap: then continue from the oldest bar received
        cap = int(bulk or getattr(provider, "_max_bars", 800))
        cursor = min(r[0] for r in rows) if rows and len(rows) >= cap else start_s
    return added


__all__ = ["BACKFILL_DAYS", "EMPTY_WINDOW_CODES", "M1Store", "StoredSeries", "backfill_history", "fetch_updates"]

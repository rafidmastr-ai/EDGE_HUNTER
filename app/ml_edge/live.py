"""Experimental live EDGE ML signals and paper trades (never changes the app's main signal).

Every 15 minutes (after each M15 close) the scheduler:
1. extends the local M1 store of the six symbols from the live provider;
2. rebuilds the features and runs every enabled model (variant A: intraday, B1: hold
   up to 12 h) over the last ``history_days`` so its self-calibrating threshold is defined;
3. records each new BUY/SELL decision as a paper trade and re-simulates all paper trades
   with the same execution rules as the research backtest.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.ml_edge.data import cost_price, swap_price
from app.ml_edge.features import build_all
from app.ml_edge.live_store import M1Store, fetch_updates
from app.ml_edge.model import SymbolModel
from app.ml_edge.walkforward import execution_config
from app.research.intraday import make_orders, simulate

logger = logging.getLogger("edge_hunter.edge_ml")
SYMBOLS = ("AUDUSD", "EURJPY", "EURUSD", "GBPUSD", "NZDUSD", "XAUUSD")
VARIANTS = {
    "A": {"session_start_hour": 7, "title_ar": "داخل اليوم (خروج خلال 4 ساعات أو 20:45 UTC)"},
    "B1": {"session_start_hour": 1, "title_ar": "حتى الهدف/الوقف، حد أقصى 12 ساعة، بدون عطلة نهاية الأسبوع"},
}
ACTIVITY_FEATURES = ("tick_volume_z", "spread_z")
FRESH_SECONDS = 20 * 60
RESIMULATE_SECONDS = 2 * 86400
EXIT_REASONS = {1: "TP", -1: "SL", 2: "SL"}
WARNING_AR = "تجريبي — إشارات بحثية للتداول الورقي فقط، لا تؤثر على الإشارة الرئيسية وليست توصية تداول."


@dataclass
class LoadedModel:
    variant: str
    symbol: str
    model: SymbolModel

    @property
    def key(self) -> str:
        return f"{self.variant}/{self.symbol}"


@dataclass
class LatestDecision:
    bar_close_utc: str | None = None
    direction: str = "NONE"
    forecast_r: float | None = None
    atr: float | None = None
    reference_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    fresh: bool = False


@dataclass
class ServiceState:
    last_refresh_utc: str | None = None
    last_error: str | None = None
    store_last_bar_utc: dict[str, str | None] = field(default_factory=dict)
    decisions: dict[str, LatestDecision] = field(default_factory=dict)


def market_open(now: datetime) -> bool:
    """FX/gold week: Sunday 21:00 UTC to Friday 21:00 UTC (no provider calls outside it)."""
    now = now.astimezone(timezone.utc)
    weekday, hour = now.weekday(), now.hour
    return not (weekday == 5 or (weekday == 6 and hour < 21) or (weekday == 4 and hour >= 21))


def _iso(seconds: int | float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).isoformat()


class EdgeMLService:
    def __init__(self, models_dir: Path, store: M1Store, database=None, provider=None, *, history_days: int = 80,
                 fetch_pause_seconds: float = 20.0, pause=time.sleep) -> None:
        self.models_dir = Path(models_dir)
        self.store = store
        self.database = database
        self.provider = provider
        self.history_days = history_days
        self.fetch_pause_seconds = fetch_pause_seconds
        self.pause = pause
        self.models: list[LoadedModel] = []
        self.load_errors: dict[str, str] = {}
        self.state = ServiceState()
        self._refresh_lock = threading.Lock()
        self._paper_start: int | None = None
        perf = self.models_dir / "performance.json"
        self.performance = json.loads(perf.read_text(encoding="utf-8")).get("models", {}) if perf.exists() else {}
        self._loaded = False
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ models
    def ensure_loaded(self) -> None:
        """Load the enabled models on first use (keeps app start-up and tests fast)."""
        with self._load_lock:
            if not self._loaded:
                self._load_models()
                self._loaded = True

    def _load_models(self) -> None:
        for variant in VARIANTS:
            base = self.models_dir / variant
            if not base.is_dir():
                continue
            for folder in sorted(p for p in base.iterdir() if p.is_dir()):
                key = f"{variant}/{folder.name}"
                try:
                    model = SymbolModel.load(folder)
                except Exception as exc:  # corrupt / tampered file: keep the service running without it
                    self.load_errors[key] = str(exc)[:200]
                    logger.warning("edge_ml model %s not loaded: %s", key, exc)
                    continue
                if model.enabled:
                    self.models.append(LoadedModel(variant, model.symbol, model))

    # ------------------------------------------------------------------ refresh
    def refresh(self, now: datetime | None = None, *, fetch: bool = True) -> None:
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
            if fetch and self.provider is not None and market_open(now):
                for k, symbol in enumerate(SYMBOLS):
                    if k:  # keep the shared provider rate limit free for user analyses
                        self.pause(self.fetch_pause_seconds)
                    fetch_updates(self.store, self.provider, symbol, now, pause=self.pause, pause_seconds=self.fetch_pause_seconds)
            self._evaluate(now)
            self.state.last_error = None
        except Exception as exc:  # the scheduler must survive any single failure
            logger.exception("edge_ml refresh failed")
            self.state.last_error = type(exc).__name__
        finally:
            self.state.last_refresh_utc = datetime.now(timezone.utc).isoformat()
            self._refresh_lock.release()

    def _evaluate(self, now: datetime) -> None:
        self.ensure_loaded()
        if not self.models:
            return
        now_s = int(now.timestamp())
        data = {s: self.store.series(s).to_symbol_data(s) for s in SYMBOLS}
        self.state.store_last_bar_utc = {s: _iso(d.m1.open_time[-1]) if len(d.m1) else None for s, d in data.items()}
        if any(len(d.m1) == 0 for d in data.values()):
            raise RuntimeError("edge_ml store is empty for at least one symbol")
        if self._paper_start is None:
            self._paper_start = self._load_paper_start(now_s)
        for variant, cfg in VARIANTS.items():
            models = [m for m in self.models if m.variant == variant]
            if not models:
                continue
            frames = build_all(data, session_start_hour=cfg["session_start_hour"])
            for loaded in models:
                frame = frames[loaded.symbol]
                x = frame.x.copy()
                for name in ACTIVITY_FEATURES:  # the live feed has no tick volume / spread: neutral value
                    col = frame.names.index(name)
                    x[np.isnan(x[:, col]), col] = 0.0
                keep = frame.tradable & (frame.close_time <= now_s) & (frame.close_time >= now_s - self.history_days * 86400)
                if not keep.any():
                    continue
                times, xs, atr = frame.close_time[keep], x[keep], frame.atr[keep]
                direction, forecast = loaded.model.decide(times, xs, atr)
                self._update_latest(loaded, data[loaded.symbol], times, direction, forecast, atr, now_s)
                self._record_paper(loaded, data[loaded.symbol], times, direction, atr, now_s)

    def _update_latest(self, loaded: LoadedModel, symbol_data, times, direction, forecast, atr, now_s: int) -> None:
        i = len(times) - 1
        d = int(direction[i])
        price = float(symbol_data.m1.close[np.searchsorted(symbol_data.m1.open_time, times[i], "left") - 1])
        dist = loaded.model.label.barrier_atr * float(atr[i])
        self.state.decisions[loaded.key] = LatestDecision(
            bar_close_utc=_iso(times[i]),
            direction={1: "BUY", -1: "SELL"}.get(d, "NONE"),
            forecast_r=round(float(forecast[i]), 4),
            atr=float(atr[i]),
            reference_price=price,
            stop_loss=price - d * dist if d else None,
            take_profit=price + d * dist if d else None,
            fresh=bool(now_s - int(times[i]) <= FRESH_SECONDS),
        )

    # ------------------------------------------------------------------ paper trades
    def _load_paper_start(self, now_s: int) -> int:
        if self.database is None:
            return now_s - 15 * 60
        row = self.database.execute("SELECT MIN(signal_time) AS first FROM edge_ml_paper_trades").fetchone()
        first = row["first"] if row is not None else None
        return int(first) if first is not None else now_s - 15 * 60

    def _resimulate_from(self, model_key: str, now_s: int) -> int:
        """Earliest signal still worth re-simulating: open trades and anything from the last two days.

        Older closed trades are frozen, so a shifting 80-day window can never rewrite them."""
        recent = now_s - RESIMULATE_SECONDS
        row = self.database.execute(
            "SELECT MIN(signal_time) AS first FROM edge_ml_paper_trades WHERE model_key = ? AND (status = 'open' OR exit_time >= ?)",
            (model_key, recent),
        ).fetchone()
        first = row["first"] if row is not None and row["first"] is not None else recent
        return max(self._paper_start, min(int(first), recent))

    def _record_paper(self, loaded: LoadedModel, symbol_data, times, direction, atr, now_s: int) -> None:
        if self.database is None:
            return
        since = self._resimulate_from(loaded.key, now_s)
        sel = (direction != 0) & (times >= since)
        n = int(sel.sum())
        label = loaded.model.label
        orders = make_orders(times[sel], direction[sel], np.full(n, np.nan), np.full(n, np.nan),
                             stop_distance=label.barrier_atr * atr[sel], target_rr=np.ones(n))
        trades = simulate(symbol_data.m1, orders, execution_config(loaded.symbol, label))
        last_bar = int(symbol_data.m1.open_time[-1])
        with self.database.transaction() as conn:
            # signals skipped because a previous paper trade was still open are not trades
            conn.execute("DELETE FROM edge_ml_paper_trades WHERE model_key = ? AND signal_time >= ?", (loaded.key, since))
            for i in range(len(trades)):
                closed = int(trades.exit_time[i]) < last_bar  # an exit on the newest bar may just be the data's end
                d = int(trades.direction[i])
                entry, stop = float(trades.entry[i]), float(trades.stop[i])
                conn.execute(
                    """
                    INSERT INTO edge_ml_paper_trades(model_key, variant, symbol, signal_time, direction, entry_time, entry, stop_loss,
                        take_profit, exit_time, exit, exit_reason, r_gross, r_net, swap_nights, status, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                    """,
                    (loaded.key, loaded.variant, loaded.symbol, int(trades.signal_time[i]), "BUY" if d > 0 else "SELL",
                     int(trades.entry_time[i]), entry, stop, entry + d * abs(entry - stop),
                     int(trades.exit_time[i]) if closed else None, float(trades.exit[i]) if closed else None,
                     EXIT_REASONS.get(int(trades.outcome[i]), "TIME") if closed else None,
                     float(trades.r_gross[i]) if closed else None, float(trades.r_net[i]) if closed else None,
                     int(trades.swap_nights[i]) if closed and trades.swap_nights is not None else 0,
                     "closed" if closed else "open"),
                )

    def paper_trades(self, symbol: str | None = None, limit: int = 200) -> list[dict]:
        if self.database is None:
            return []
        query = "SELECT * FROM edge_ml_paper_trades"
        args: tuple = ()
        if symbol:
            query += " WHERE symbol = ?"
            args = (symbol.upper(),)
        query += " ORDER BY signal_time DESC LIMIT ?"
        rows = self.database.execute(query, (*args, int(limit))).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            for k in ("signal_time", "entry_time", "exit_time"):
                item[k] = _iso(item[k]) if item.get(k) is not None else None
            out.append(item)
        return out

    def paper_summary(self, model_key: str) -> dict:
        if self.database is None:
            return {"trades": 0}
        rows = self.database.execute(
            "SELECT r_net FROM edge_ml_paper_trades WHERE model_key = ? AND status = 'closed'", (model_key,)).fetchall()
        open_count = self.database.execute(
            "SELECT COUNT(*) AS n FROM edge_ml_paper_trades WHERE model_key = ? AND status = 'open'", (model_key,)).fetchone()["n"]
        r = np.array([row["r_net"] for row in rows], dtype=float)
        if not len(r):
            return {"closed_trades": 0, "open_trades": int(open_count)}
        return {"closed_trades": int(len(r)), "open_trades": int(open_count), "win_rate": round(float((r > 0).mean()), 4),
                "avg_r_net": round(float(r.mean()), 4), "total_r_net": round(float(r.sum()), 2)}

    # ------------------------------------------------------------------ status
    def status(self, symbol: str | None = None) -> dict:
        self.ensure_loaded()
        models = []
        for loaded in self.models:
            if symbol and loaded.symbol != symbol.upper():
                continue
            decision = self.state.decisions.get(loaded.key, LatestDecision())
            models.append({
                "model": loaded.key,
                "variant": loaded.variant,
                "variant_title_ar": VARIANTS[loaded.variant]["title_ar"],
                "symbol": loaded.symbol,
                "architecture": loaded.model.architecture,
                "latest": decision.__dict__,
                "research_results": self.performance.get(loaded.key, {}),
                "paper": self.paper_summary(loaded.key),
                "costs": {"spread": cost_price(loaded.symbol),
                          "swap_per_night": swap_price(loaded.symbol) if loaded.model.label.hold else 0.0},
            })
        return {
            "enabled": True,
            "experimental": True,
            "warning_ar": WARNING_AR,
            "models": models,
            "models_loaded": len(self.models),
            "load_errors": self.load_errors,
            "last_refresh_utc": self.state.last_refresh_utc,
            "last_error": self.state.last_error,
            "store_last_bar_utc": self.state.store_last_bar_utc,
        }


class EdgeMLScheduler:
    """Background thread: refresh 90 s after every M15 close (or every N x 15 min) and once at start."""

    def __init__(self, service: EdgeMLService, *, interval_minutes: int = 15, offset_seconds: int = 90) -> None:
        self.service = service
        self.interval_seconds = max(15, int(interval_minutes)) // 15 * 15 * 60
        self.offset_seconds = offset_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="edge-ml-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        self.service.refresh()
        while not self._stop.is_set():
            now = time.time()
            next_run = (int(now) // self.interval_seconds + 1) * self.interval_seconds + self.offset_seconds
            if self._stop.wait(max(1.0, next_run - now)):
                break
            self.service.refresh()


__all__ = ["EdgeMLScheduler", "EdgeMLService", "LatestDecision", "SYMBOLS", "VARIANTS", "WARNING_AR", "market_open"]

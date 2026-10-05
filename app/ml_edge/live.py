"""Live EDGE ML recommendations and paper trades.

The latest fresh BUY/SELL decision of a model is served as a live recommendation
(``recommendation`` / ``best_recommendation``); the user executes it manually.

Every 15 minutes (after each M15 close) the scheduler:
1. extends the local M1 store of every served symbol from the live provider;
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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.ml_edge.data import cost_price, swap_price
from app.ml_edge.features import build_all
from app.ml_edge.data import TRAINING_SYMBOLS
from app.ml_edge.live_store import M1Store, backfill_history, fetch_updates
from app.ml_edge.model import SymbolModel
from app.ml_edge.walkforward import execution_config
from app.research.intraday import make_orders, simulate

logger = logging.getLogger("edge_hunter.edge_ml")
# The six training symbols plus the yen crosses served by the EURJPY model (cross features use only the six).
SYMBOLS = ("AUDUSD", "EURJPY", "EURUSD", "GBPUSD", "NZDUSD", "XAUUSD", "AUDJPY", "CADJPY")
VARIANTS = {
    "A": {"session_start_hour": 7, "title_ar": "داخل اليوم (خروج خلال 4 ساعات أو 20:45 UTC)",
          "exit_ar": "اخرج عند الهدف أو الوقف، أو بعد 4 ساعات، أو عند 20:45 UTC أيهما أسبق"},
    "B1": {"session_start_hour": 1, "title_ar": "حتى الهدف/الوقف، حد أقصى 12 ساعة، بدون عطلة نهاية الأسبوع",
           "exit_ar": "اخرج عند الهدف أو الوقف، أو بعد 12 ساعة، أو الجمعة 20:45 UTC أيهما أسبق"},
}
ACTIVITY_FEATURES = ("tick_volume_z", "spread_z")
FRESH_SECONDS = 20 * 60
RESIMULATE_SECONDS = 2 * 86400
RETRY_SECONDS = 60
MAX_RETRY_SECONDS = 30 * 60
DAILY_REQUEST_BUDGET = 600  # background requests per UTC day; Twelve Data free plan = 800 credits/day
MIN_HISTORY_DAYS = 85  # 60-day feature windows + the threshold's 20-day warm-up
EXIT_REASONS = {1: "TP", -1: "SL", 2: "SL"}
WARNING_AR = "توصيات من نماذج إحصائية مختبرة على بيانات سابقة — ليست ضماناً للربح؛ نفّذها يدوياً وبمخاطرة مناسبة."


def normalize_symbol(symbol: str | None) -> str | None:
    """``XAU/USD`` / ``xauusd`` -> ``XAUUSD`` (the UI and the provider use the slash form)."""
    if not symbol:
        return None
    return str(symbol).strip().upper().replace("/", "").replace(" ", "")


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
    expected_r: float | None = None  # model forecast in the trade direction, minus the spread in R
    # the model's own leaning even when it is below its tested entry threshold (direction NONE)
    candidate_direction: str | None = None
    candidate_stop_loss: float | None = None
    candidate_take_profit: float | None = None
    candidate_expected_r: float | None = None


@dataclass
class ServiceState:
    last_refresh_utc: str | None = None
    last_error: str | None = None
    store_last_bar_utc: dict[str, str | None] = field(default_factory=dict)
    decisions: dict[str, LatestDecision] = field(default_factory=dict)
    refreshing: bool = False
    backfilling: str | None = None  # symbol whose history is being fetched from the provider
    stage: str | None = None  # what the running refresh is doing (shown in the UI)
    failures: int = 0  # consecutive failed refreshes (scheduler back-off)
    evaluated_utc: str | None = None  # last successful evaluation
    store_coverage: dict[str, dict] = field(default_factory=dict)


def market_open(now: datetime) -> bool:
    """FX/gold week: Sunday 21:00 UTC to Friday 21:00 UTC (no provider calls outside it)."""
    now = now.astimezone(timezone.utc)
    weekday, hour = now.weekday(), now.hour
    return not (weekday == 5 or (weekday == 6 and hour < 21) or (weekday == 4 and hour >= 21))


def _last_market_close(now: datetime) -> int:
    """Epoch seconds of the most recent Friday 21:00 UTC at or before ``now``."""
    now = now.astimezone(timezone.utc)
    days_back = (now.weekday() - 4) % 7
    friday = (now - timedelta(days=days_back)).replace(hour=21, minute=0, second=0, microsecond=0)
    if friday > now:
        friday -= timedelta(days=7)
    return int(friday.timestamp())


def _iso(seconds: int | float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).isoformat()


class _BudgetedProvider:
    """The live provider as seen by the background refresh: counts requests per UTC day, stops at the
    daily budget (the rest is left for user analyses) and pauses until 00:00 UTC when the provider
    says the day's credits are used up."""

    def __init__(self, inner, daily_budget: int) -> None:
        self.inner = inner
        self.daily_budget = int(daily_budget)
        self.day: str | None = None
        self.used = 0
        self.paused_until: datetime | None = None

    def __getattr__(self, name):  # _request_times, _rate_limit, bulk_max_bars, _max_bars, configured ...
        return getattr(self.inner, name)

    def blocked_reason(self, now: datetime | None = None) -> str | None:
        now = now or datetime.now(timezone.utc)
        if self.paused_until is not None and now < self.paused_until:
            return "credits"
        if self.day == now.strftime("%Y-%m-%d") and self.used >= self.daily_budget:
            return "budget"
        return None

    def get_ohlc(self, symbol, timeframe, start, end, **kwargs):
        from app.providers.models import LiveProviderError

        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if self.day != today:
            self.day, self.used = today, 0
        reason = self.blocked_reason(now)
        if reason:
            raise LiveProviderError(f"edge_ml background requests paused ({reason})", code=f"edge_ml_{reason}_paused")
        self.used += 1
        try:
            return self.inner.get_ohlc(symbol, timeframe, start, end, **kwargs)
        except LiveProviderError as exc:
            if exc.code in {"provider_429", "provider_rate_limited"} and "credit" in str(exc).lower():
                self.paused_until = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
            raise


class EdgeMLService:
    def __init__(self, models_dir: Path, store: M1Store, database=None, provider=None, *, history_days: int = 80,
                 fetch_pause_seconds: float = 12.0, pause=time.sleep, fresh_seconds: int = FRESH_SECONDS,
                 daily_request_budget: int = DAILY_REQUEST_BUDGET) -> None:
        self.models_dir = Path(models_dir)
        self.store = store
        self.database = database
        self.provider = provider
        self.fetcher = _BudgetedProvider(provider, daily_request_budget) if provider is not None else None
        self.history_days = history_days
        self.fetch_pause_seconds = fetch_pause_seconds
        self.pause = pause
        self.fresh_seconds = int(fresh_seconds)
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
        self.state.refreshing = True
        try:
            live_clock = now is None  # tests pass a fixed ``now``; the scheduler uses the wall clock
            now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
            # weekends: no new bars, but missing history / the gap up to Friday's close is still fetched
            if fetch and self.fetcher is not None and (market_open(now) or self._needs_catch_up(now)):
                self._fetch_all(now)
                if live_clock and (datetime.now(timezone.utc) - now).total_seconds() > 60:
                    # a long catch-up (first start): top up the minutes that passed meanwhile, then
                    # evaluate at the current time so the newest decisions are fresh, not already stale
                    now = datetime.now(timezone.utc)
                    self._fetch_all(now)
            if live_clock:
                now = datetime.now(timezone.utc)
            self.state.stage = "حساب قرارات النماذج"
            self._evaluate(now)
            self.state.last_error = None
            self.state.failures = 0
            self.state.evaluated_utc = datetime.now(timezone.utc).isoformat()
        except Exception as exc:  # the scheduler must survive any single failure
            logger.exception("edge_ml refresh failed")
            self.state.last_error = f"{type(exc).__name__}: {exc}"[:200]
            self.state.failures += 1
        finally:
            self.state.refreshing = False
            self.state.stage = None
            self.state.last_refresh_utc = datetime.now(timezone.utc).isoformat()
            self._refresh_lock.release()

    def _fetch_all(self, now: datetime) -> None:
        for k, symbol in enumerate(SYMBOLS):
            if self.fetcher.blocked_reason():
                return
            if k:  # keep the shared provider rate limit free for user analyses
                self.pause(self.fetch_pause_seconds)
            self.state.stage = f"جلب آخر البيانات: {symbol} ({k + 1}/{len(SYMBOLS)})"
            fetch_updates(self.store, self.fetcher, symbol, now, pause=self.pause, pause_seconds=self.fetch_pause_seconds)
            # no repository CSV history: fetch the ~95 days the models need from the provider (once)
            if self._coverage_days(symbol) < MIN_HISTORY_DAYS:
                self.state.stage = f"جلب تاريخ 95 يوماً (M15): {symbol} ({k + 1}/{len(SYMBOLS)})"
                self.state.backfilling = symbol
                try:
                    backfill_history(self.store, self.fetcher, symbol, now, pause=self.pause, pause_seconds=self.fetch_pause_seconds)
                finally:
                    self.state.backfilling = None

    def _needs_catch_up(self, now: datetime) -> bool:
        last_close = _last_market_close(now)
        for symbol in SYMBOLS:
            series = self.store.series(symbol)
            if not len(series.t) or self._coverage_days(symbol) < MIN_HISTORY_DAYS or series.t[-1] < last_close - 3600:
                return True
        return False

    def _coverage_days(self, symbol: str) -> float:
        series = self.store.series(symbol)
        return float(series.t[-1] - series.t[0]) / 86400 if len(series.t) else 0.0

    def _history_ok(self, symbol: str) -> bool:
        """Enough M1 history for the model's features (own symbol + the six cross-market symbols)."""
        cov = self.state.store_coverage
        if not cov:  # not measured yet (tests inject decisions directly)
            return True
        return all(cov.get(s, {}).get("days", 0) >= MIN_HISTORY_DAYS for s in {*TRAINING_SYMBOLS, symbol} if s in cov)

    def _evaluate(self, now: datetime) -> None:
        self.ensure_loaded()
        if not self.models:
            return
        now_s = int(now.timestamp())
        data = {s: self.store.series(s).to_symbol_data(s) for s in SYMBOLS}
        self.state.store_last_bar_utc = {s: _iso(d.m1.open_time[-1]) if len(d.m1) else None for s, d in data.items()}
        self.state.store_coverage = {
            s: {"first_bar_utc": _iso(d.m1.open_time[0]) if len(d.m1) else None,
                "last_bar_utc": _iso(d.m1.open_time[-1]) if len(d.m1) else None,
                "days": round(float(d.m1.open_time[-1] - d.m1.open_time[0]) / 86400, 1) if len(d.m1) else 0.0}
            for s, d in data.items()
        }
        empty = [s for s, d in data.items() if len(d.m1) == 0]
        if empty:
            raise RuntimeError(f"no M1 history for {', '.join(empty)}")
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
        lean = 1 if forecast[i] >= 0 else -1
        self.state.decisions[loaded.key] = LatestDecision(
            bar_close_utc=_iso(times[i]),
            direction={1: "BUY", -1: "SELL"}.get(d, "NONE"),
            forecast_r=round(float(forecast[i]), 4),
            atr=float(atr[i]),
            reference_price=price,
            stop_loss=price - d * dist if d else None,
            take_profit=price + d * dist if d else None,
            fresh=bool(now_s - int(times[i]) <= self.fresh_seconds),
            expected_r=round(float(d * forecast[i] - loaded.model.cost_price / dist), 4) if d else None,
            candidate_direction="BUY" if lean > 0 else "SELL",
            candidate_stop_loss=price - lean * dist,
            candidate_take_profit=price + lean * dist,
            candidate_expected_r=round(float(abs(forecast[i]) - loaded.model.cost_price / dist), 4),
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
            args = (normalize_symbol(symbol),)
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
    # ------------------------------------------------------------------ recommendations
    def _ranked(self, now: datetime | None = None, *, include_candidates: bool = False, allow_no_edge: bool = False) -> list[dict]:
        """Fresh model decisions at ``now``: tier "active" (passed the tested entry threshold) first,
        then - if asked - tier "below_threshold" (the model's leaning, not validated), each by expected R."""
        self.ensure_loaded()
        now_s = int((now or datetime.now(timezone.utc)).timestamp())
        out = []
        for loaded in self.models:
            d = self.state.decisions.get(loaded.key)
            if d is None or d.bar_close_utc is None or d.reference_price is None or not self._history_ok(loaded.symbol):
                continue
            bar_s = int(datetime.fromisoformat(d.bar_close_utc).timestamp())
            if now_s - bar_s > self.fresh_seconds or bar_s > now_s:
                continue
            if d.direction != "NONE" and d.stop_loss is not None:
                tier, direction, stop, take, expected = "active", d.direction, d.stop_loss, d.take_profit, d.expected_r
            elif include_candidates and d.candidate_direction and (allow_no_edge or (d.candidate_expected_r or 0) > 0):
                tier, direction, stop, take, expected = ("below_threshold", d.candidate_direction, d.candidate_stop_loss,
                                                         d.candidate_take_profit, d.candidate_expected_r)
            else:
                continue
            out.append({
                "symbol": loaded.symbol,
                "model": loaded.key,
                "variant": loaded.variant,
                "variant_title_ar": VARIANTS[loaded.variant]["title_ar"],
                "exit_rule_ar": VARIANTS[loaded.variant]["exit_ar"],
                "tier": tier,
                "direction": direction,
                "entry": d.reference_price,
                "stop_loss": stop,
                "take_profit": take,
                "risk_reward": 1.0,
                "atr_m15": d.atr,
                "bar_close_utc": d.bar_close_utc,
                "age_minutes": round((now_s - bar_s) / 60, 1),
                "forecast_r": d.forecast_r,
                "expected_r": expected,
                "research_results": self.performance.get(loaded.key, {}),
                "costs": {"spread": cost_price(loaded.symbol),
                          "swap_per_night": swap_price(loaded.symbol) if loaded.model.label.hold else 0.0},
            })
        return sorted(out, key=lambda r: (r["tier"] != "active", -(r["expected_r"] or 0.0)))

    def recommendation(self, symbol: str | None, now: datetime | None = None, *, include_candidates: bool = True) -> dict | None:
        """Best recommendation for one symbol: an active signal if any (highest expected R if A and B1
        both signal), otherwise the model's fresh leaning (tier "below_threshold")."""
        symbol = normalize_symbol(symbol)
        # a typed pair always gets its model's direction, even a near-zero leaning (reported as no edge)
        ranked = self._ranked(now, include_candidates=include_candidates, allow_no_edge=include_candidates)
        return next((r for r in ranked if r["symbol"] == symbol), None)

    def model_symbols(self) -> list[str]:
        self.ensure_loaded()
        return sorted({m.symbol for m in self.models})

    def best_recommendation(self, now: datetime | None = None) -> tuple[dict | None, list[dict]]:
        """(best recommendation across all models, the full ranked list).

        Active signals come first; when none is active the strongest fresh below-threshold
        candidate is returned (tier "below_threshold"), so a fresh model state always yields one.
        """
        ranked = self._ranked(now, include_candidates=True)
        return (ranked[0] if ranked else None), ranked

    def readiness(self, now: datetime | None = None) -> dict:
        """Why there may be no recommendation, in plain Arabic (for the UI and /api/edge-ml)."""
        self.ensure_loaded()
        now = now or datetime.now(timezone.utc)
        now_s = int(now.timestamp())
        problems = []
        if not self.models:
            problems.append("لم يُحمَّل أي نموذج من models/edge_ml — تأكد من نسخ مجلد models من GitHub.")
        if self.state.evaluated_utc is None:
            if self.provider is None or not getattr(self.provider, "configured", False):
                problems.append("مزوّد البيانات الحية غير مفعّل، فلا تُحدَّث النماذج (شغّل التطبيق بوضع live مع مفتاح Twelve Data).")
            elif self.state.refreshing or self.state.last_refresh_utc is None:
                stage = f" — الآن: {self.state.stage}" if self.state.stage else ""
                problems.append(f"النماذج قيد التحميل والتحديث الأول (عادة 3–10 دقائق){stage}.")
        if self.state.last_error:
            problems.append(f"آخر تحديث للنماذج فشل: {self.state.last_error}")
        short = {s: c.get("days", 0) for s, c in self.state.store_coverage.items() if c.get("days", 0) < MIN_HISTORY_DAYS}
        if short:
            detail = "، ".join(f"{s} {d:.0f} يوماً" for s, d in short.items())
            problems.append(f"تاريخ M1 غير كافٍ ({detail}) من {MIN_HISTORY_DAYS} يوماً مطلوبة — يجري جلبه تلقائياً من Twelve Data "
                            "(طلبان تقريباً لكل زوج، مرة واحدة).")
        blocked = self.fetcher.blocked_reason(now) if self.fetcher is not None else None
        if blocked == "credits":
            problems.append("نفد رصيد Twelve Data اليومي — توقف تحديث النماذج حتى 00:00 UTC (03:00 بتوقيت بغداد). "
                            "قلّل EDGE_HUNTER_EDGE_ML_REFRESH_MINUTES أو استخدم خطة أعلى.")
        elif blocked == "budget":
            problems.append(f"بلغ تحديث النماذج حده اليومي ({self.fetcher.daily_budget} طلب) — يُترك الباقي لتحليلاتك حتى 00:00 UTC.")
        if market_open(now) and self.state.store_last_bar_utc:
            lagging = [s for s, t in self.state.store_last_bar_utc.items()
                       if t and now_s - int(datetime.fromisoformat(t).timestamp()) > 3 * 3600]
            if lagging:
                problems.append(f"البيانات الحية متأخرة لـ {', '.join(lagging)} (تحقق من مفتاح Twelve Data وحدود الطلبات).")
        if not market_open(now):
            problems.append("السوق مغلق الآن (عطلة نهاية الأسبوع) — لا توجد توصيات حتى افتتاح السوق.")
        fresh = [k for k, d in self.state.decisions.items() if d.bar_close_utc
                 and now_s - int(datetime.fromisoformat(d.bar_close_utc).timestamp()) <= self.fresh_seconds]
        return {"ready": bool(fresh), "fresh_models": len(fresh), "refreshing": self.state.refreshing,
                "evaluated_utc": self.state.evaluated_utc, "problems_ar": problems,
                "provider_requests_today": self.fetcher.used if self.fetcher is not None else 0,
                "daily_request_budget": self.fetcher.daily_budget if self.fetcher is not None else 0}

    def quote_to_usd(self, symbol: str) -> float | None:
        """USD value of one unit of the symbol's quote currency, from the latest stored closes."""
        symbol = normalize_symbol(symbol) or ""
        quote = symbol[-3:]
        if quote == "USD":
            return 1.0
        if quote == "JPY":
            try:
                eurjpy = float(self.store.series("EURJPY").c[-1])
                eurusd = float(self.store.series("EURUSD").c[-1])
            except (IndexError, KeyError):
                return None
            return eurusd / eurjpy if eurjpy > 0 else None
        return None

    def closest_symbol(self) -> str:
        """Symbol whose model is nearest to a signal (largest |forecast|), EURJPY when nothing is known."""
        best, value = "EURJPY", -1.0
        for loaded in self.models:
            d = self.state.decisions.get(loaded.key)
            if d is not None and d.forecast_r is not None and abs(d.forecast_r) > value:
                best, value = loaded.symbol, abs(d.forecast_r)
        return best

    def status(self, symbol: str | None = None, now: datetime | None = None) -> dict:
        self.ensure_loaded()
        symbol = normalize_symbol(symbol)
        models = []
        for loaded in self.models:
            if symbol and loaded.symbol != symbol:
                continue
            decision = self.state.decisions.get(loaded.key, LatestDecision())
            latest = dict(decision.__dict__)
            if decision.bar_close_utc:  # freshness at request time, not at refresh time
                age = (now or datetime.now(timezone.utc)) - datetime.fromisoformat(decision.bar_close_utc)
                latest["fresh"] = 0 <= age.total_seconds() <= self.fresh_seconds
            models.append({
                "model": loaded.key,
                "variant": loaded.variant,
                "variant_title_ar": VARIANTS[loaded.variant]["title_ar"],
                "symbol": loaded.symbol,
                "architecture": loaded.model.architecture,
                "latest": latest,
                "research_results": self.performance.get(loaded.key, {}),
                "paper": self.paper_summary(loaded.key),
                "costs": {"spread": cost_price(loaded.symbol),
                          "swap_per_night": swap_price(loaded.symbol) if loaded.model.label.hold else 0.0},
            })
        return {
            "enabled": True,
            "experimental": True,
            "symbol": symbol,
            "fresh_minutes": round(self.fresh_seconds / 60),
            "warning_ar": WARNING_AR,
            "models": models,
            "models_loaded": len(self.models),
            "load_errors": self.load_errors,
            "last_refresh_utc": self.state.last_refresh_utc,
            "last_error": self.state.last_error,
            "store_last_bar_utc": self.state.store_last_bar_utc,
            "store_coverage": self.state.store_coverage,
            "readiness": self.readiness(now),
        }


class EdgeMLScheduler:
    """Background thread: refresh 90 s after every M15 close (or every N x 15 min) and once at start."""

    def __init__(self, service: EdgeMLService, *, interval_minutes: int = 30, offset_seconds: int = 90) -> None:
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
            if self.service.state.evaluated_utc is None or self.service.state.last_error:
                # not ready yet / failed: retry soon, backing off 1, 2, 4 ... 30 minutes
                backoff = min(MAX_RETRY_SECONDS, RETRY_SECONDS * 2 ** max(0, self.service.state.failures - 1))
                next_run = min(next_run, now + backoff)
            if self._stop.wait(max(1.0, next_run - now)):
                break
            self.service.refresh()


__all__ = ["EdgeMLScheduler", "EdgeMLService", "LatestDecision", "SYMBOLS", "VARIANTS", "WARNING_AR", "market_open", "normalize_symbol"]

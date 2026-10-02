"""Market analysis service backing the web UI.

Data sources:
- ``live`` (default): OHLC comes exclusively from the configured live provider
  (Twelve Data). Local CSV files are never read in this mode; a provider failure
  is reported as ``LiveDataUnavailableError`` instead of silently switching data.
- ``local``: explicit offline mode that reads ``data/raw/*.csv``. Those CSVs exist
  for backtesting/learning and deterministic tests, not as a live data source.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from app.data.cleaning import drop_synthetic_flat_runs
from app.data.schema import CanonicalOHLC
from app.providers.live_provider import LiveMarketDataProvider
from app.providers.models import LiveProviderError
from app.features.engine import FeatureEngine
from app.signals.costs import CostGate
from app.signals.engine import SignalConfidenceEngine, SignalFilter
from app.signals.models import FinalSignalDecision
from app.strategies.models import SignalState, StrategyContext, StrategySignal
from app.strategies.registry import StrategyRegistry
from app.web.schemas import AnalyzeRequest, CapitalImpact, MarketCandle


TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
}

# The UI intentionally does not expose timeframe selection. The service evaluates
# a compact multi-timeframe set and uses M15 as the chart/primary context.
ANALYSIS_TIMEFRAMES = ("M5", "M15", "H1")
PRIMARY_TIMEFRAME = "M15"
CHART_CANDLES = 120
BASE_HISTORY_BARS = 15_000

# Conservative symbol contract sizes used only for transparent lot/P&L arithmetic.
# They are not provider execution specifications.
FOREX_CODES = {"EUR", "GBP", "USD", "JPY", "CHF", "AUD", "CAD", "NZD", "SGD", "HKD", "NOK", "SEK", "DKK", "PLN", "TRY", "ZAR", "MXN", "CNH", "CNY"}
METAL_CODES = {"XAU", "XAG", "XPT", "XPD"}


class DataUnavailableError(RuntimeError):
    """Raised when no usable market-data source exists for a symbol."""


class LiveDataUnavailableError(DataUnavailableError):
    """Raised when live mode is enabled but the provider cannot supply data."""

    def __init__(self, message: str, *, code: str = "live_data_unavailable") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SymbolDataset:
    symbol: str
    path: Path
    bars: tuple[CanonicalOHLC, ...]


class LocalOHLCAnalysisService:
    """Load OHLC (live provider or explicit local CSV mode) and call the signal engine."""

    def __init__(
        self,
        data_root: Path | None = None,
        *,
        live_provider: LiveMarketDataProvider | None = None,
        data_mode: str = "local",
        live_history_bars: int = 800,
        signal_filter: SignalFilter | None = None,
        cost_gate: CostGate | None = None,
        edge_ml=None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_root = data_root or (self.project_root / "data" / "raw")
        self.feature_engine = FeatureEngine()
        # Optional learned per-strategy filter (read-only inference of approved models).
        self.signal_filter = signal_filter
        # Optional cost-aware gate: withholds setups whose trading cost is too large vs the stop.
        self.cost_gate = cost_gate
        # Optional experimental EDGE ML status (paper signals; never changes the decision above).
        self.edge_ml = edge_ml
        self.signal_engine = SignalConfidenceEngine(StrategyRegistry.default(), signal_filter=signal_filter, cost_gate=cost_gate)
        self.live_provider = live_provider
        self.data_mode = data_mode if data_mode in {"local", "live"} else "local"
        self.live_history_bars = max(50, min(int(live_history_bars), 5000))
        self._dataset_cache: dict[str, SymbolDataset] = {}

    def analyze(self, request: AnalyzeRequest) -> dict:
        return self.analyze_with_observations(request)[0]

    def analyze_with_observations(
        self,
        request: AnalyzeRequest,
    ) -> tuple[dict, list[tuple[str, StrategySignal, dict]]]:
        """Analysis with live EDGE ML recommendations taking priority over the classic engine.

        - symbol empty (None): the best active model recommendation across all symbols;
          if no model is signalling, NO_CLEAR_SIGNAL on the symbol nearest to a signal.
        - symbol with an active model signal: the model recommendation is the main result.
        - otherwise: the classic multi-strategy analysis, unchanged.
        """
        auto = request.symbol is None
        recommendation, ranked = None, []
        if auto:
            if self.edge_ml is None:
                raise ValueError("symbol is required when EDGE ML is disabled")
            recommendation, ranked = self.edge_ml.best_recommendation()
            symbol = recommendation["symbol"] if recommendation else self.edge_ml.closest_symbol()
            request = request.model_copy(update={"symbol": symbol})
        elif self.edge_ml is not None:
            try:
                recommendation = self.edge_ml.recommendation(request.symbol)
            except Exception:  # the model layer must never break the classic analysis
                recommendation = None
        result, observations = self._analyze_classic(request)
        if auto:
            result["metadata"]["edge_ml"] = self.edge_ml_status(None)
            result["metadata"]["edge_ml_ranked"] = ranked
        if recommendation is not None:
            self._apply_model_recommendation(result, request, recommendation, auto=auto)
        elif auto:
            self._apply_no_recommendation(result)
        else:
            result["metadata"]["recommendation_source"] = "classic"
            self._add_model_hint(result, request.symbol)
        return result, observations

    def _apply_model_recommendation(self, result: dict, request: AnalyzeRequest, rec: dict, *, auto: bool) -> None:
        classic = {key: result.get(key) for key in ("direction", "confidence", "confidence_label_ar", "selected_strategy",
                                                     "selected_variant", "entry", "target", "stop_loss")}
        forward = rec["research_results"].get("new_forward", {})
        win_rate = forward.get("win_rate")
        confidence = round(100.0 * win_rate, 1) if win_rate is not None else 50.0
        label = "قوي" if confidence >= 65 else "متوسط" if confidence >= 55 else "ضعيف"
        below = rec.get("tier") == "below_threshold"
        if below:  # the model's leaning only: its quality was never validated
            confidence, label = round(min(confidence, 50.0) / 2, 1), "ضعيف"
        entry, stop, target = rec["entry"], rec["stop_loss"], rec["take_profit"]
        lot_size, impact = self._model_lot_and_impact(request, entry, stop, target)
        bar_utc = datetime.fromisoformat(rec["bar_close_utc"])
        bar_baghdad = bar_utc.astimezone(timezone(timedelta(hours=3)))
        reasons = [
            (f"أفضل توصية متاحة الآن — دون عتبة الدخول المختبرة لنموذج EDGE ML ({rec['model']}): "
             + ("لا يوجد أي نموذج تجاوز عتبته الآن، فهذا أقوى ميل بين النماذج" if auto
                else "نموذج هذا الزوج لم يتجاوز عتبته الآن، فهذا ميله الحالي")
             + " وجودته غير مثبتة في الاختبارات.")
            if below else f"توصية نموذج EDGE ML ({rec['model']}) — {rec['variant_title_ar']}.",
            f"القرار عند إغلاق شمعة M15 الساعة {bar_utc:%H:%M} UTC ({bar_baghdad:%H:%M} بتوقيت بغداد)، قبل {rec['age_minutes']:.0f} دقيقة.",
            f"العائد المتوقع للنموذج: {rec['expected_r']:+.2f}R بعد السبريد (نسبة الهدف إلى الوقف 1:1، الوقف والهدف = 2 × ATR على M15).",
            f"الخروج: {rec['exit_rule_ar']}.",
            "الدخول المرجعي هو آخر سعر إغلاق؛ ادخل بسعر السوق الحالي وحافظ على نفس مسافة الوقف والهدف.",
        ]
        if forward:
            reasons.append(
                f"نتائج الاختبار الأمامي للنموذج: {forward.get('trades')} صفقة، فوز {100 * (forward.get('win_rate') or 0):.0f}%، "
                f"متوسط {forward.get('avg_r', 0):+.2f}R، PF {forward.get('profit_factor')}."
            )
        reasons.append("توصية من نموذج إحصائي وليست ضماناً — نفّذها يدوياً وبمخاطرة مناسبة.")
        result.update({
            "status": "success",
            "direction": rec["direction"],
            "confidence": confidence,
            "confidence_label_ar": label,
            "selected_strategy": f"EDGE ML {rec['variant']}",
            "selected_variant": rec["model"],
            "entry": entry,
            "target": target,
            "stop_loss": stop,
            "risk_reward": 1.0,
            "lot_size": lot_size,
            "capital_impact": impact,
            "reasons": reasons,
        })
        result["metadata"]["recommendation_source"] = "edge_ml_auto" if auto else "edge_ml"
        result["metadata"]["recommendation_tier"] = rec.get("tier", "active")
        result["metadata"]["model_recommendation"] = rec
        result["metadata"]["classic_analysis"] = classic

    def _add_model_hint(self, result: dict, symbol: str) -> None:
        """A classic "no clear signal" on a pair without a usable model: say where model trades are."""
        if self.edge_ml is None:
            return
        try:
            pairs = self.edge_ml.model_symbols()
            readiness = self.edge_ml.readiness()
        except Exception:
            return
        result["metadata"]["edge_ml_model_symbols"] = pairs
        if result.get("direction") != "NO_CLEAR_SIGNAL":
            return
        compact = (symbol or "").replace("/", "").upper()
        if compact in pairs:  # a model pair without a fresh decision: explain why
            result["reasons"] = [*result.get("reasons", []), "نموذج EDGE ML لهذا الزوج ليس لديه قرار حديث الآن.",
                                 *readiness.get("problems_ar", [])]
        else:
            result["reasons"] = [*result.get("reasons", []),
                                 f"لا يوجد نموذج EDGE ML لهذا الرمز؛ نماذج التوصيات متاحة لـ: {', '.join(pairs)}.",
                                 "اترك حقل الرمز فارغاً للحصول على أفضل صفقة متاحة من النماذج."]

    def _apply_no_recommendation(self, result: dict) -> None:
        try:
            readiness = self.edge_ml.readiness()
        except Exception:
            readiness = {"problems_ar": []}
        problems = readiness.get("problems_ar") or [
            "لا توجد قرارات حديثة من النماذج؛ تُحدَّث بعد إغلاق كل شمعة M15 — أعد المحاولة بعد دقائق."]
        classic = {key: result.get(key) for key in ("direction", "confidence", "selected_strategy", "entry", "target", "stop_loss")}
        result.update({
            "status": "no_clear_signal",
            "direction": "NO_CLEAR_SIGNAL",
            "confidence": 0.0,
            "confidence_label_ar": "لا توجد توصية",
            "selected_strategy": None,
            "selected_variant": None,
            "entry": None,
            "target": None,
            "stop_loss": None,
            "risk_reward": None,
            "capital_impact": None,
            "reasons": [
                "لا توجد توصية من نماذج EDGE ML الآن.",
                *problems,
                f"الرسم يعرض {result['symbol']}.",
            ],
        })
        result["metadata"]["edge_ml_readiness"] = readiness
        result["metadata"]["recommendation_source"] = "edge_ml_auto"
        result["metadata"]["classic_analysis"] = classic

    def _model_lot_and_impact(self, request: AnalyzeRequest, entry: float, stop: float, target: float):
        """Lot and money figures with the risk converted to USD (JPY-quoted pairs use the live USDJPY)."""
        if request.lot_mode == "manual":
            lot = round(request.lot_size or 0.01, 2)
        else:
            lot = 0.01
        quote_usd = self.edge_ml.quote_to_usd(request.symbol) if self.edge_ml is not None else None
        if request.capital is None or quote_usd is None:
            return lot, None
        contract = _contract_size_for_symbol(request.symbol)
        risk_amount = request.capital * (request.risk_percent / 100.0)
        usd_per_lot_at_stop = abs(entry - stop) * contract * quote_usd
        if request.lot_mode != "manual" and usd_per_lot_at_stop > 0:
            lot = round(math.floor(max(0.01, min(100.0, risk_amount / usd_per_lot_at_stop)) * 100) / 100, 2)
        impact = CapitalImpact(
            target_profit=round(abs(target - entry) * contract * quote_usd * lot, 2),
            stop_loss_loss=round(usd_per_lot_at_stop * lot, 2),
            risk_amount=round(risk_amount, 2),
        )
        return lot, impact.model_dump()

    def _analyze_classic(
        self,
        request: AnalyzeRequest,
    ) -> tuple[dict, list[tuple[str, StrategySignal, dict]]]:
        """Return the analysis plus the actionable strategy setups it observed.

        Observations are (timeframe, raw strategy setup annotated with the learned
        probability, decision-time feature values). The web layer stores them
        only after the response has been sent, for later outcome labelling.
        """
        observations: list[tuple[str, StrategySignal, dict]] = []
        dataset = self._load_dataset(request.symbol) if self.data_mode == "local" else None
        live_source_used = False
        timeframe_results: dict[str, FinalSignalDecision] = {}
        timeframe_bars: dict[str, list[CanonicalOHLC]] = {}

        for timeframe in ANALYSIS_TIMEFRAMES:
            if self.data_mode == "live":
                # Live analysis uses provider data only. There is deliberately no
                # CSV fallback: mixing sources would present stale/local prices
                # as live and hide the real provider error.
                try:
                    bars = self._load_live_bars(request.symbol, timeframe)
                    live_source_used = True
                except LiveProviderError as exc:
                    raise LiveDataUnavailableError(
                        "live market data is currently unavailable",
                        code=exc.code,
                    ) from exc
            else:
                assert dataset is not None
                bars = self._resample(dataset.bars, timeframe)
            if len(bars) < 10:
                continue
            symbol_name = request.symbol.upper()
            analysis = self.feature_engine.compute(bars, symbol_name, timeframe)
            context = StrategyContext.from_series(bars, analysis)
            raw_signals, signals = self.signal_engine.evaluate_signals(context)
            timeframe_results[timeframe] = self.signal_engine.decide(context, signals)
            timeframe_bars[timeframe] = bars
            for raw, final in zip(raw_signals, signals):
                if raw.state != SignalState.SIGNAL:
                    continue
                learned = {key: value for key, value in dict(final.metadata).items() if str(key).startswith("ml_")}
                annotated = replace(raw, metadata={**dict(raw.metadata), **learned}) if learned else raw
                observations.append((timeframe, annotated, dict(context.current.values)))

        if not timeframe_results:
            if self.data_mode == "live":
                raise LiveDataUnavailableError(
                    f"insufficient live market data for {request.symbol} in required analysis timeframes",
                    code="live_insufficient_data",
                )
            raise DataUnavailableError(
                f"insufficient historical data for {request.symbol} in required analysis timeframes"
            )

        decision, selected_tf = self._select_multitimeframe_decision(timeframe_results)
        chart_bars = timeframe_bars.get(selected_tf, timeframe_bars.get(PRIMARY_TIMEFRAME, []))
        chart = [
            MarketCandle(
                timestamp=bar.timestamp.isoformat(),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume) if bar.volume is not None else None,
            )
            for bar in chart_bars[-CHART_CANDLES:]
        ]

        lot_size = self._resolve_lot_size(request, decision)
        capital_impact = self._capital_impact(request, decision, lot_size)
        comparisons = [self._comparison(tf, result) for tf, result in timeframe_results.items()]

        response = decision.to_dict()
        response.update(
            {
                "status": "success" if decision.is_trade_signal else "no_clear_signal",
                "symbol": request.symbol.upper(),
                "timestamp": decision.timestamp.isoformat(),
                "time_zone": "UTC+3 / Baghdad",
                "primary_timeframe": selected_tf,
                "analyzed_timeframes": list(timeframe_results),
                "lot_size": lot_size,
                "strategy_comparison": comparisons,
                "chart": [item.model_dump() for item in chart],
                "capital_impact": capital_impact.model_dump() if capital_impact else None,
                "metadata": {
                    **dict(decision.metadata),
                    "data_source": (self.live_provider.name if live_source_used and self.live_provider else "local_csv"),
                    "data_mode": self.data_mode,
                    # Kept for response compatibility; live mode never falls back.
                    "data_fallback_used": False,
                    "data_window_bars": len(dataset.bars) if dataset is not None else sum(len(items) for items in timeframe_bars.values()),
                    "multitimeframe_selection": "deterministic_agreement_then_confidence",
                    "timeframe_results": {
                        tf: {
                            "direction": result.direction.value,
                            "confidence": result.confidence,
                        }
                        for tf, result in timeframe_results.items()
                    },
                    "financial_values_hidden_without_capital": request.capital is None,
                    "live_provider_health": self.live_health(),
                    "strategy_learning": self.strategy_learning_status(),
                    "cost_gate": self._cost_gate_summary(timeframe_results),
                    "trade_cost": _selected_trade_cost(decision),
                    "edge_ml": self.edge_ml_status(request.symbol),
                },
            }
        )
        return response, observations

    def _cost_gate_summary(self, results: dict[str, FinalSignalDecision]) -> dict:
        if self.cost_gate is None:
            return {"enabled": False}
        withheld = [
            {"timeframe": tf, "strategy": item.strategy_name, "cost_r": item.signal.metadata.get("cost_r")}
            for tf, result in results.items()
            for item in result.strategy_evaluations
            if item.signal.metadata.get("cost_decision") == "rejected"
        ]
        return {
            "enabled": True,
            "max_cost_r": self.cost_gate.max_cost_r,
            "withheld_setups": withheld,
            "note": "setups whose estimated spread+slippage+commission exceeds max_cost_r of the stop distance are withheld",
        }

    def edge_ml_status(self, symbol: str) -> dict:
        if self.edge_ml is None:
            return {"enabled": False}
        try:
            return self.edge_ml.status(symbol)
        except Exception:  # experimental add-on: never break the main analysis
            return {"enabled": True, "models": [], "last_error": "edge_ml_status_failed"}

    def strategy_learning_status(self) -> dict:
        status = getattr(self.signal_filter, "status", None)
        if not callable(status):
            return {"enabled": False, "model_loaded": False, "active_strategies": []}
        return status()

    def _load_live_bars(self, symbol: str, timeframe: str) -> list[CanonicalOHLC]:
        if self.live_provider is None:
            raise LiveProviderError(
                "live market data provider is not configured",
                code="live_provider_unconfigured",
                status_code=503,
            )
        end = datetime.now(timezone.utc)
        minutes = TIMEFRAME_MINUTES[timeframe]
        start = end - timedelta(minutes=minutes * (self.live_history_bars + 5))
        provider_bars = self.live_provider.get_ohlc(symbol, timeframe, start, end)
        # Never build a signal from a still-forming candle. The timestamp of a
        # provider intraday bar denotes its opening bucket, so anything in the
        # current bucket is excluded from the decision window.
        current_bucket = _floor_timeframe(end, minutes)
        closed_bars = [bar for bar in provider_bars if bar.timestamp < current_bucket]
        return [
            CanonicalOHLC(
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            for bar in closed_bars
        ]

    def live_health(self) -> dict:
        if self.live_provider is None:
            return {
                "provider": "none",
                "configured": False,
                "status": "unconfigured",
            }
        return self.live_provider.health().to_dict()

    def _load_dataset(self, symbol: str) -> SymbolDataset:
        symbol = symbol.upper()
        cached = self._dataset_cache.get(symbol)
        if cached is not None:
            return cached

        compact = symbol.replace("/", "")
        candidates = [self.data_root / f"{symbol}.csv", self.data_root / f"{compact}.csv"]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not path.exists():
            matches = [
                candidate
                for candidate in self.data_root.glob("*.csv")
                if candidate.stem.upper() in {symbol.upper(), compact.upper()}
            ]
            if matches:
                path = matches[0]
        if not path.exists():
            raise DataUnavailableError(f"no local CSV dataset found for {symbol}")

        bars = tuple(self._read_csv(path, limit=BASE_HISTORY_BARS))
        if len(bars) < 10:
            raise DataUnavailableError(f"CSV dataset is too small for {symbol}")
        dataset = SymbolDataset(symbol, path, bars)
        self._dataset_cache[symbol] = dataset
        return dataset

    @staticmethod
    def _read_csv(path: Path, limit: int | None = None) -> Iterable[CanonicalOHLC]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DataUnavailableError(f"CSV has no header: {path}")
            aliases = {name.strip().lower(): name for name in reader.fieldnames}
            timestamp_key = next((aliases[name] for name in ("timestamp", "datetime", "time", "date") if name in aliases), None)
            if timestamp_key is None:
                raise DataUnavailableError(f"CSV has no timestamp column: {path}")

            keys: dict[str, str] = {}
            for field in ("open", "high", "low", "close"):
                key = aliases.get(field)
                if key is None:
                    raise DataUnavailableError(f"CSV missing {field}: {path}")
                keys[field] = key

            volume_key = aliases.get("volume") or aliases.get("tick_volume") or aliases.get("vol")
            from collections import deque

            tail: deque[tuple[datetime, CanonicalOHLC]] = deque(maxlen=limit or 0) if limit else deque()
            for row in reader:
                if not row:
                    continue
                try:
                    timestamp = _parse_timestamp(row.get(timestamp_key, ""))
                    values = {field: Decimal(str(row.get(key, "")).strip()) for field, key in keys.items()}
                    volume = None
                    if volume_key and str(row.get(volume_key, "")).strip():
                        volume = Decimal(str(row[volume_key]).strip())
                    bar = CanonicalOHLC(timestamp=timestamp, volume=volume, **values)
                except (ArithmeticError, ValueError, TypeError, DataUnavailableError):
                    continue
                if any(not value.is_finite() or value <= 0 for value in values.values()):
                    continue
                if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close, bar.high):
                    continue
                if limit:
                    tail.append((timestamp, bar))
                else:
                    tail.append((timestamp, bar))

            ordered = {}
            for timestamp, bar in tail:
                ordered.setdefault(timestamp, bar)
            # Closed-market filler (flat candles repeating the last close) is not market data.
            cleaned, _ = drop_synthetic_flat_runs([ordered[timestamp] for timestamp in sorted(ordered)])
            yield from cleaned

    @staticmethod
    def _resample(bars: tuple[CanonicalOHLC, ...] | list[CanonicalOHLC], timeframe: str) -> list[CanonicalOHLC]:
        minutes = TIMEFRAME_MINUTES[timeframe]
        if minutes == 1:
            return list(bars)

        buckets: dict[int, list[CanonicalOHLC]] = {}
        for bar in bars:
            epoch_minute = int(bar.timestamp.timestamp() // 60)
            bucket = epoch_minute - (epoch_minute % minutes)
            buckets.setdefault(bucket, []).append(bar)

        result: list[CanonicalOHLC] = []
        for bucket, group in sorted(buckets.items()):
            group = sorted(group, key=lambda item: item.timestamp)
            volumes = [item.volume for item in group]
            volume = sum(volumes, Decimal("0")) if volumes and all(item is not None for item in volumes) else None
            result.append(
                CanonicalOHLC(
                    timestamp=datetime.fromtimestamp(bucket * 60, tz=timezone.utc),
                    open=group[0].open,
                    high=max(item.high for item in group),
                    low=min(item.low for item in group),
                    close=group[-1].close,
                    volume=volume,
                )
            )
        return result

    @staticmethod
    def _select_multitimeframe_decision(
        results: dict[str, FinalSignalDecision],
    ) -> tuple[FinalSignalDecision, str]:
        actionable = [(tf, decision) for tf, decision in results.items() if decision.is_trade_signal]
        if not actionable:
            preferred = results.get(PRIMARY_TIMEFRAME) or next(iter(results.items()))[1]
            return preferred, next(tf for tf, decision in results.items() if decision is preferred)

        buy = [(tf, decision) for tf, decision in actionable if decision.direction.value == "BUY"]
        sell = [(tf, decision) for tf, decision in actionable if decision.direction.value == "SELL"]
        groups = [("BUY", buy), ("SELL", sell)]
        groups.sort(key=lambda item: (len(item[1]), max((decision.confidence for _, decision in item[1]), default=0.0)), reverse=True)
        direction, group = groups[0]
        opposing_count = len(groups[1][1])
        required = 2 if len(results) >= 3 else max(1, (len(results) + 1) // 2)
        if len(group) < required or opposing_count >= len(group):
            fallback = results.get(PRIMARY_TIMEFRAME) or group[0][1]
            fallback_tf = PRIMARY_TIMEFRAME if PRIMARY_TIMEFRAME in results else group[0][0]
            # Convert conflicting multi-timeframe actionable outputs into NO_CLEAR_SIGNAL
            if len(group) < required or opposing_count:
                return _neutralize_decision(fallback), fallback_tf
        chosen_tf, chosen = max(group, key=lambda item: (item[1].confidence, item[0] == PRIMARY_TIMEFRAME))
        return chosen, chosen_tf

    @staticmethod
    def _resolve_lot_size(request: AnalyzeRequest, decision: FinalSignalDecision) -> float | None:
        if request.lot_mode == "manual":
            return round(request.lot_size or 0.01, 2)
        if request.capital is None or not decision.is_trade_signal or decision.entry is None or decision.stop_loss is None:
            return 0.01
        distance = abs(float(decision.entry) - float(decision.stop_loss))
        if distance <= 0:
            return 0.01
        contract_size = _contract_size_for_symbol(request.symbol)
        risk_amount = request.capital * (request.risk_percent / 100.0)
        units = risk_amount / (distance * contract_size)
        lots = max(0.01, min(100.0, units))
        return round(math.floor(lots * 100) / 100, 2)

    @staticmethod
    def _capital_impact(
        request: AnalyzeRequest,
        decision: FinalSignalDecision,
        lot_size: float | None,
    ) -> CapitalImpact | None:
        if request.capital is None or not decision.is_trade_signal or lot_size is None:
            return None
        contract_size = _contract_size_for_symbol(request.symbol)
        risk_amount = request.capital * (request.risk_percent / 100.0)
        target_profit = 0.0
        stop_loss_loss = 0.0
        if decision.entry is not None and decision.target is not None:
            target_profit = abs(float(decision.target) - float(decision.entry)) * contract_size * lot_size
        if decision.entry is not None and decision.stop_loss is not None:
            stop_loss_loss = abs(float(decision.entry) - float(decision.stop_loss)) * contract_size * lot_size
        return CapitalImpact(
            target_profit=round(target_profit, 2),
            stop_loss_loss=round(stop_loss_loss, 2),
            risk_amount=round(risk_amount, 2),
        )

    @staticmethod
    def _comparison(timeframe: str, decision: FinalSignalDecision) -> dict:
        representative = {
            "timeframe": timeframe,
            "direction": decision.direction.value,
            "confidence": round(decision.confidence, 2),
            "label_ar": decision.confidence_label_ar,
            "strategy": decision.selected_strategy,
            "variant": decision.selected_variant,
        }
        return representative


def _selected_trade_cost(decision: FinalSignalDecision) -> dict | None:
    """Cost annotation of the setup behind the final decision (None if no trade / no gate)."""
    if not decision.is_trade_signal:
        return None
    for item in decision.strategy_evaluations:
        if item.strategy_name == decision.selected_strategy and item.variant == decision.selected_variant:
            meta = item.signal.metadata
            if "cost_r" not in meta:
                return None
            keys = ("cost_estimate", "cost_basis", "cost_r", "cost_max_r", "min_stop_distance", "net_risk_reward_after_cost")
            return {key: meta.get(key) for key in keys}
    return None


def _floor_timeframe(value: datetime, minutes: int) -> datetime:
    """Start of the UTC bucket of ``minutes`` length that contains ``value``."""
    epoch_minute = int(value.astimezone(timezone.utc).timestamp() // 60)
    return datetime.fromtimestamp((epoch_minute - epoch_minute % minutes) * 60, tz=timezone.utc)


def _split_symbol(symbol: str) -> tuple[str, str]:
    text = symbol.upper().replace(" ", "")
    if "/" in text:
        base, _, quote = text.partition("/")
        return base, quote
    return text[:3], text[3:]


def _contract_size_for_symbol(symbol: str) -> float:
    """Units per 1.00 lot, used only for transparent lot/P&L arithmetic.

    Metals use their common spot contract (oz), Forex the standard 100,000
    units lot, and crypto/other instruments are quoted per 1 unit of the base.
    """
    base, quote = _split_symbol(symbol)
    metal_contracts = {"XAU": 100.0, "XAG": 5000.0, "XPT": 100.0, "XPD": 100.0}
    if base in METAL_CODES:
        return metal_contracts[base]
    if base in FOREX_CODES and quote in FOREX_CODES:
        return 100_000.0
    return 1.0


def _parse_timestamp(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise DataUnavailableError("empty timestamp")
    if text.isdigit():
        number = int(text)
        if number > 10_000_000_000:
            return datetime.fromtimestamp(number / 1000.0, tz=timezone.utc)
        if number > 1_000_000_000:
            return datetime.fromtimestamp(number, tz=timezone.utc)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DataUnavailableError(f"invalid timestamp: {text!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _neutralize_decision(decision: FinalSignalDecision) -> FinalSignalDecision:
    from app.signals.models import FinalSignalDirection

    return FinalSignalDecision(
        timestamp=decision.timestamp,
        symbol=decision.symbol,
        timeframe=decision.timeframe,
        direction=FinalSignalDirection.NO_CLEAR_SIGNAL,
        confidence=0.0,
        confidence_label_ar="ضعيف",
        entry=None,
        stop_loss=None,
        target=None,
        risk_reward=None,
        selected_strategy=None,
        selected_variant=None,
        reasons=("multi_timeframe_conflict_or_insufficient_agreement",),
        strategy_evaluations=decision.strategy_evaluations,
        directional_support=decision.directional_support,
        scoring_components={},
        metadata={
            **dict(decision.metadata),
            "multi_timeframe_neutralized": True,
        },
    )

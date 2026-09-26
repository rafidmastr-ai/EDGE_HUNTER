"""Real local-OHLC analysis service backing the Phase 09 web UI."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from app.data.schema import CanonicalOHLC
from app.providers.live_provider import LiveMarketDataProvider
from app.providers.models import LiveProviderError
from app.features.engine import FeatureEngine
from app.signals.engine import SignalConfidenceEngine
from app.signals.models import FinalSignalDecision
from app.strategies.models import StrategyContext
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
    """Load local CSVs, resample OHLC and call the Phase 08 signal engine."""

    def __init__(
        self,
        data_root: Path | None = None,
        *,
        live_provider: LiveMarketDataProvider | None = None,
        data_mode: str = "local",
        live_history_bars: int = 800,
        live_fallback_to_local: bool = True,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_root = data_root or (self.project_root / "data" / "raw")
        self.feature_engine = FeatureEngine()
        self.signal_engine = SignalConfidenceEngine(StrategyRegistry.default())
        self.live_provider = live_provider
        self.data_mode = data_mode if data_mode in {"local", "live"} else "local"
        self.live_history_bars = max(50, min(int(live_history_bars), 5000))
        self.live_fallback_to_local = bool(live_fallback_to_local)
        self._dataset_cache: dict[str, SymbolDataset] = {}

    def analyze(self, request: AnalyzeRequest) -> dict:
        dataset = self._load_dataset(request.symbol) if self.data_mode == "local" else None
        live_source_used = False
        timeframe_results: dict[str, FinalSignalDecision] = {}
        timeframe_bars: dict[str, list[CanonicalOHLC]] = {}

        for timeframe in ANALYSIS_TIMEFRAMES:
            if self.data_mode == "live":
                try:
                    bars = self._load_live_bars(request.symbol, timeframe)
                    live_source_used = True
                except LiveProviderError as exc:
                    if not self.live_fallback_to_local:
                        raise LiveDataUnavailableError(
                            "live market data is currently unavailable",
                            code=exc.code,
                        ) from exc
                    if dataset is None:
                        dataset = self._load_dataset(request.symbol)
                    bars = self._resample(dataset.bars, timeframe)
            else:
                assert dataset is not None
                bars = self._resample(dataset.bars, timeframe)
            if len(bars) < 10:
                continue
            symbol_name = request.symbol.upper()
            analysis = self.feature_engine.compute(bars, symbol_name, timeframe)
            context = StrategyContext.from_series(bars, analysis)
            timeframe_results[timeframe] = self.signal_engine.analyze(context)
            timeframe_bars[timeframe] = bars

        if not timeframe_results:
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
                    "data_fallback_used": self.data_mode == "live" and not live_source_used,
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
                },
            }
        )
        return response

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
            for timestamp in sorted(ordered):
                yield ordered[timestamp]

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

"""Reusable OHLC feature engine independent of trading strategy logic."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from typing import Sequence

from app.data.schema import CanonicalOHLC
from app.features.calculations import (
    atr_wilder,
    close_returns,
    ema,
    log_returns,
    macd,
    rolling_std,
    rsi_wilder,
)
from app.features.models import FeatureDefinition, MarketAnalysisSeries, MarketAnalysisSnapshot


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters for deterministic feature generation."""

    ema_periods: tuple[int, ...] = (20, 50, 200)
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    volatility_period: int = 20
    volume_period: int = 20
    return_periods: tuple[int, ...] = (1, 5, 20)

    def __post_init__(self) -> None:
        periods = (*self.ema_periods, self.rsi_period, self.atr_period, self.macd_fast_period,
                   self.macd_slow_period, self.macd_signal_period, self.volatility_period,
                   self.volume_period, *self.return_periods)
        if not all(isinstance(period, int) and period > 0 for period in periods):
            raise ValueError("all feature periods must be positive integers")
        if not self.ema_periods:
            raise ValueError("ema_periods cannot be empty")
        if self.macd_fast_period >= self.macd_slow_period:
            raise ValueError("macd fast period must be smaller than slow period")

    @property
    def warmup_bars_required(self) -> int:
        return max(
            *self.ema_periods,
            self.rsi_period + 1,
            self.atr_period,
            self.macd_slow_period + self.macd_signal_period - 1,
            self.volatility_period,
            *(period + 1 for period in self.return_periods),
        )


class FeatureEngine:
    """Compute reusable OHLC-derived analysis features.

    The engine expects canonical timestamps to represent the analysis observation
    timestamp. Each output at T uses only bars whose positions are <= T.
    """

    VERSION = "phase03-v1"

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def compute(
        self,
        bars: Sequence[CanonicalOHLC],
        symbol: str,
        timeframe: str,
    ) -> MarketAnalysisSeries:
        records = list(bars)
        self._validate_input(records)
        closes = [float(bar.close) for bar in records]

        ema_values = {period: ema(closes, period) for period in self.config.ema_periods}
        rsi_values = rsi_wilder(closes, self.config.rsi_period)
        atr_values = atr_wilder(records, self.config.atr_period)
        macd_line, macd_signal, macd_histogram = macd(
            closes,
            self.config.macd_fast_period,
            self.config.macd_slow_period,
            self.config.macd_signal_period,
        )
        log_return_values = log_returns(closes)
        rolling_volatility = rolling_std(log_return_values[1:], self.config.volatility_period)
        trailing_returns = {
            period: close_returns(closes, period) for period in self.config.return_periods
        }
        volume_enabled = self._valid_volume_available(records)
        volume_sma = self._volume_sma(records) if volume_enabled else [None] * len(records)

        definitions = self._build_definitions(volume_enabled)
        snapshots: list[MarketAnalysisSnapshot] = []
        previous_close: float | None = None
        warmup_required = self.config.warmup_bars_required

        for index, bar in enumerate(records):
            close = closes[index]
            bar_open = float(bar.open)
            high = float(bar.high)
            low = float(bar.low)
            bar_range = high - low
            body = abs(close - bar_open)
            upper_wick = max(0.0, high - max(bar_open, close))
            lower_wick = max(0.0, min(bar_open, close) - low)
            values: dict[str, float | int | str | bool | None] = {
                "price.open": bar_open,
                "price.high": high,
                "price.low": low,
                "price.close": close,
                "structure.range": bar_range,
                "structure.body": body,
                "structure.body_ratio": (body / bar_range) if bar_range > 0 else None,
                "structure.upper_wick": upper_wick,
                "structure.lower_wick": lower_wick,
                "structure.close_location": ((close - low) / bar_range) if bar_range > 0 else None,
                "structure.direction": (
                    "BULLISH" if close > bar_open else "BEARISH" if close < bar_open else "NEUTRAL"
                ),
                "returns.log_1": log_return_values[index],
                "trend.ema_alignment": self._ema_alignment(index, ema_values, close),
                "trend.ema_spread_20_50_pct": self._spread_pct(
                    self._value_at(ema_values.get(20), index),
                    self._value_at(ema_values.get(50), index),
                    close,
                ),
                "momentum.rsi": rsi_values[index],
                "momentum.macd": macd_line[index],
                "momentum.macd_signal": macd_signal[index],
                "momentum.macd_histogram": macd_histogram[index],
                "volatility.atr": atr_values[index],
                "volatility.atr_pct": self._atr_pct(atr_values[index], close),
                "volatility.realized_log_std": self._rolling_vol_at_index(
                    rolling_volatility, index
                ),
                "warmup.bars_required": warmup_required,
                "warmup.bars_available": index + 1,
                "warmup.complete": index + 1 >= warmup_required,
            }

            for period, values_for_period in ema_values.items():
                values[f"trend.ema_{period}"] = values_for_period[index]
            for period, returns_for_period in trailing_returns.items():
                values[f"returns.simple_{period}"] = returns_for_period[index]

            if volume_enabled:
                volume = float(bar.volume)  # validated by _valid_volume_available
                values["volume.value"] = volume
                values[f"volume.sma_{self.config.volume_period}"] = volume_sma[index]
                sma_value = volume_sma[index]
                values["volume.ratio"] = (volume / sma_value) if sma_value else None
            else:
                values["volume.value"] = None
                values[f"volume.sma_{self.config.volume_period}"] = None
                values["volume.ratio"] = None

            metadata = {
                "engine_version": self.VERSION,
                "feature_count": len(values),
                "volume_features_enabled": volume_enabled,
                "timestamp_policy": "uses_observations_at_or_before_timestamp",
                "previous_close": previous_close,
                "source": "canonical_ohlc",
            }
            snapshot = MarketAnalysisSnapshot(
                timestamp=bar.timestamp,
                symbol=symbol.upper(),
                timeframe=timeframe,
                values=values,
                metadata=metadata,
                warmup_complete=index + 1 >= warmup_required,
            )
            snapshots.append(snapshot)
            previous_close = close

        return MarketAnalysisSeries(
            symbol=symbol.upper(),
            timeframe=timeframe,
            snapshots=tuple(snapshots),
            definitions=tuple(definitions),
            warmup_bars_required=warmup_required,
            volume_features_enabled=volume_enabled,
            engine_version=self.VERSION,
        )

    def _build_definitions(self, volume_enabled: bool) -> list[FeatureDefinition]:
        definitions = [
            FeatureDefinition("price.open", "price", "Canonical open price", "OHLC", 1),
            FeatureDefinition("price.high", "price", "Canonical high price", "OHLC", 1),
            FeatureDefinition("price.low", "price", "Canonical low price", "OHLC", 1),
            FeatureDefinition("price.close", "price", "Canonical close price", "OHLC", 1),
            FeatureDefinition("structure.range", "structure", "High-low range", "OHLC", 1),
            FeatureDefinition("structure.body", "structure", "Absolute candle body", "OHLC", 1),
            FeatureDefinition("structure.body_ratio", "structure", "Body divided by range", "OHLC", 1),
            FeatureDefinition("structure.upper_wick", "structure", "Upper wick size", "OHLC", 1),
            FeatureDefinition("structure.lower_wick", "structure", "Lower wick size", "OHLC", 1),
            FeatureDefinition("structure.close_location", "structure", "Close location within candle range", "OHLC", 1),
            FeatureDefinition("structure.direction", "structure", "Bullish, bearish or neutral candle", "OHLC", 1),
            FeatureDefinition("returns.log_1", "returns", "One-bar logarithmic return", "Close", 2),
            FeatureDefinition("trend.ema_alignment", "trend", "EMA ordering state", "Close", max(self.config.ema_periods)),
            FeatureDefinition("trend.ema_spread_20_50_pct", "trend", "EMA 20/50 spread as percent of close", "Close", 50),
            FeatureDefinition("momentum.rsi", "momentum", "Wilder RSI", "Close", self.config.rsi_period + 1),
            FeatureDefinition("momentum.macd", "momentum", "MACD line", "Close", self.config.macd_slow_period),
            FeatureDefinition("momentum.macd_signal", "momentum", "MACD signal line", "Close", self.config.macd_slow_period + self.config.macd_signal_period - 1),
            FeatureDefinition("momentum.macd_histogram", "momentum", "MACD histogram", "Close", self.config.macd_slow_period + self.config.macd_signal_period - 1),
            FeatureDefinition("volatility.atr", "volatility", "Wilder ATR", "OHLC", self.config.atr_period),
            FeatureDefinition("volatility.atr_pct", "volatility", "ATR as percent of close", "OHLC", self.config.atr_period),
            FeatureDefinition("volatility.realized_log_std", "volatility", "Trailing population std of one-bar log returns", "Close", self.config.volatility_period + 1),
        ]
        definitions.extend(
            FeatureDefinition(
                f"trend.ema_{period}", "trend", f"EMA {period}", "Close", period
            )
            for period in self.config.ema_periods
        )
        definitions.extend(
            FeatureDefinition(
                f"returns.simple_{period}", "returns", f"Simple return over {period} bars", "Close", period + 1
            )
            for period in self.config.return_periods
        )
        if volume_enabled:
            definitions.extend(
                [
                    FeatureDefinition("volume.value", "volume", "Validated source volume", "Volume", 1, True),
                    FeatureDefinition(f"volume.sma_{self.config.volume_period}", "volume", "Volume SMA", "Volume", self.config.volume_period, True),
                    FeatureDefinition("volume.ratio", "volume", "Current volume divided by volume SMA", "Volume", self.config.volume_period, True),
                ]
            )
        return definitions

    @staticmethod
    def _validate_input(records: Sequence[CanonicalOHLC]) -> None:
        previous = None
        for index, bar in enumerate(records):
            if not bar.timestamp:
                raise ValueError(f"missing timestamp at index {index}")
            if previous is not None and bar.timestamp <= previous:
                raise ValueError("feature engine requires strictly increasing timestamps")
            previous = bar.timestamp
            prices = (bar.open, bar.high, bar.low, bar.close)
            if any(not isinstance(value, Decimal) for value in prices):
                raise TypeError("canonical OHLC prices must be Decimal instances")
            if any(not value.is_finite() or value <= 0 for value in prices):
                raise ValueError(f"invalid OHLC price at index {index}")
            if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close, bar.high):
                raise ValueError(f"impossible OHLC relationship at index {index}")

    @staticmethod
    def _valid_volume_available(records: Sequence[CanonicalOHLC]) -> bool:
        if not records:
            return False
        return all(
            volume is not None and volume.is_finite() and volume >= 0
            for volume in (bar.volume for bar in records)
        )

    def _volume_sma(self, records: Sequence[CanonicalOHLC]) -> list[Optional[float]]:
        volumes = [float(bar.volume) for bar in records]  # type: ignore[arg-type]
        output: list[Optional[float]] = [None] * len(volumes)
        period = self.config.volume_period
        if len(volumes) < period:
            return output
        window_sum = sum(volumes[:period])
        output[period - 1] = window_sum / period
        for index in range(period, len(volumes)):
            window_sum += volumes[index] - volumes[index - period]
            output[index] = window_sum / period
        return output

    @staticmethod
    def _value_at(values: list[Optional[float]] | None, index: int) -> Optional[float]:
        return None if values is None else values[index]

    @staticmethod
    def _spread_pct(first: Optional[float], second: Optional[float], close: float) -> Optional[float]:
        if first is None or second is None or close == 0:
            return None
        return ((first - second) / close) * 100.0

    @staticmethod
    def _atr_pct(atr: Optional[float], close: float) -> Optional[float]:
        if atr is None or close == 0:
            return None
        return (atr / close) * 100.0

    @staticmethod
    def _rolling_vol_at_index(values: list[Optional[float]], index: int) -> Optional[float]:
        source_index = index - 1
        return values[source_index] if 0 <= source_index < len(values) else None

    @staticmethod
    def _ema_alignment(
        index: int,
        ema_values: dict[int, list[Optional[float]]],
        close: float,
    ) -> str:
        available = [values[index] for values in ema_values.values() if values[index] is not None]
        if len(available) < 2:
            return "WARMUP"
        periods = sorted(ema_values)
        ordered = [ema_values[period][index] for period in periods]
        if any(value is None for value in ordered):
            return "WARMUP"
        if all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)) and close > ordered[0]:  # type: ignore[operator]
            return "BULLISH"
        if all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1)) and close < ordered[0]:  # type: ignore[operator]
            return "BEARISH"
        return "MIXED"

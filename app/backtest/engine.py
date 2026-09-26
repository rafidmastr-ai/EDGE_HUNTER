"""Candle/event-based backtesting engine for EDGE HUNTER Phase 05."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Sequence

from app.backtest.config import BacktestConfig, IntrabarPolicy
from app.backtest.metrics import calculate_metrics
from app.backtest.models import BacktestResult, TradeOutcome, TradeRecord
from app.data.schema import CanonicalOHLC
from app.features.engine import FeatureEngine
from app.features.models import MarketAnalysisSeries
from app.strategies.models import SignalDirection, SignalState, Strategy, StrategyContext, StrategySignal


class BacktestEngine:
    """Reusable single-strategy candle/event backtester.

    A strategy is evaluated on bar ``i`` using only history through bar ``i``.
    Phase 04 defines entry at the decision-bar close, so the trade is opened at
    that close and TP/SL monitoring starts on bar ``i + 1``. Signals arriving
    while another trade is open are ignored; the policy is intentionally
    explicit for the Phase 05 measurement instrument and can be generalized in
    a later research phase.
    """

    VERSION = "phase05-v1"

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: Strategy,
        bars: Sequence[CanonicalOHLC],
        analysis: MarketAnalysisSeries | None = None,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> BacktestResult:
        """Run one deterministic backtest over canonical OHLC bars."""
        records = tuple(bars)
        self._validate_bars(records)
        if not records:
            raise ValueError("backtest requires at least one OHLC bar")

        if analysis is None:
            if symbol is None or timeframe is None:
                raise ValueError("symbol and timeframe are required when analysis is not supplied")
            analysis_series = FeatureEngine().compute(records, symbol=symbol, timeframe=timeframe)
        else:
            analysis_series = analysis
            if len(analysis_series.snapshots) != len(records):
                raise ValueError("analysis series length must match OHLC bars")
            if symbol is not None and analysis_series.symbol.upper() != symbol.upper():
                raise ValueError("analysis symbol does not match supplied symbol")
            if timeframe is not None and analysis_series.timeframe != timeframe:
                raise ValueError("analysis timeframe does not match supplied timeframe")

        run_symbol = (symbol or analysis_series.symbol).upper()
        run_timeframe = timeframe or analysis_series.timeframe
        trades: list[TradeRecord] = []
        open_trade: _OpenTrade | None = None
        current_equity = self.config.starting_capital

        for index, bar in enumerate(records):
            if open_trade is not None:
                completed = self._process_open_trade(open_trade, bar, index)
                if completed is not None:
                    trades.append(completed)
                    if (
                        current_equity is not None
                        and completed.outcome != TradeOutcome.AMBIGUOUS_SKIPPED
                    ):
                        current_equity += completed.pnl_net
                    open_trade = None

            if open_trade is not None:
                continue

            context = StrategyContext.from_series(records, analysis_series, index)
            signal = strategy.generate(context)
            self._validate_signal(signal, bar.timestamp, run_symbol, run_timeframe)
            if signal.state == SignalState.SIGNAL:
                open_trade = self._open_trade(signal, current_equity)
                open_trade.entry_index = index

        if open_trade is not None:
            # The position was opened at the final decision-bar close and has no
            # future candle available for outcome evaluation. Record it as an
            # explicit data-end expiry instead of silently dropping the signal.
            trades.append(
                self._close_at_level(
                    open_trade,
                    records[-1],
                    len(records) - 1,
                    TradeOutcome.EXPIRED,
                    float(records[-1].close),
                    "DATA_END",
                )
            )

        metrics = calculate_metrics(
            trades,
            starting_capital=self.config.starting_capital,
        )
        metadata = self._build_metadata(
            strategy,
            records,
            analysis_series,
            run_symbol,
            run_timeframe,
        )
        return BacktestResult(
            engine_version=self.VERSION,
            strategy_name=strategy.name,
            strategy_variant=strategy.variant,
            symbol=run_symbol,
            timeframe=run_timeframe,
            start_timestamp=records[0].timestamp,
            end_timestamp=records[-1].timestamp,
            trades=tuple(trades),
            metrics=metrics,
            reproducibility_metadata=metadata,
            generated_at_utc=datetime.now(timezone.utc),
        )

    def _open_trade(self, signal: StrategySignal, capital: float | None) -> "_OpenTrade":
        assert signal.entry is not None and signal.stop_loss is not None and signal.target is not None
        assert signal.direction in (SignalDirection.BUY, SignalDirection.SELL)
        quantity = self.config.position_units(signal.entry, signal.stop_loss, capital)
        price_risk = abs(signal.entry - signal.stop_loss)
        if price_risk <= 0:
            raise ValueError("signal entry and stop_loss must differ")
        planned_rr = abs(signal.target - signal.entry) / price_risk
        return _OpenTrade(
            strategy_name=signal.strategy_name,
            strategy_variant=signal.variant,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            direction=signal.direction.value,
            signal_timestamp=signal.timestamp,
            entry_timestamp=signal.timestamp,
            entry_price=signal.entry,
            stop_loss=signal.stop_loss,
            target=signal.target,
            planned_risk_reward=planned_rr,
            quantity=quantity,
            entry_index=-1,
        )

    def _process_open_trade(
        self,
        trade: "_OpenTrade",
        bar: CanonicalOHLC,
        bar_index: int,
    ) -> TradeRecord | None:
        tp_touched, sl_touched = self._touches(trade, bar)
        if tp_touched and sl_touched:
            if self.config.intrabar_policy == IntrabarPolicy.SKIP_TRADE:
                return self._close_ambiguous(trade, bar, bar_index)
            if self.config.intrabar_policy == IntrabarPolicy.TARGET_FIRST:
                return self._close_at_level(
                    trade,
                    bar,
                    bar_index,
                    TradeOutcome.WIN,
                    trade.target,
                    "TP_AND_SL_TARGET_FIRST",
                )
            return self._close_at_level(
                trade,
                bar,
                bar_index,
                TradeOutcome.LOSS,
                trade.stop_loss,
                "TP_AND_SL_STOP_FIRST",
            )

        if tp_touched:
            return self._close_at_level(trade, bar, bar_index, TradeOutcome.WIN, trade.target, "TP")
        if sl_touched:
            return self._close_at_level(trade, bar, bar_index, TradeOutcome.LOSS, trade.stop_loss, "SL")

        bars_held = bar_index - trade.entry_index
        if self.config.max_bars_in_trade is not None and bars_held >= self.config.max_bars_in_trade:
            return self._close_at_level(
                trade,
                bar,
                bar_index,
                TradeOutcome.EXPIRED,
                float(bar.close),
                "EXPIRY",
            )
        return None

    @staticmethod
    def _touches(trade: "_OpenTrade", bar: CanonicalOHLC) -> tuple[bool, bool]:
        high = float(bar.high)
        low = float(bar.low)
        if trade.direction == SignalDirection.BUY.value:
            return high >= trade.target, low <= trade.stop_loss
        return low <= trade.target, high >= trade.stop_loss

    def _close_at_level(
        self,
        trade: "_OpenTrade",
        bar: CanonicalOHLC,
        bar_index: int,
        outcome: TradeOutcome,
        requested_exit_price: float,
        reason: str,
    ) -> TradeRecord:
        if outcome == TradeOutcome.AMBIGUOUS_SKIPPED:
            raise ValueError("use _close_ambiguous for skipped ambiguous candles")

        effective_entry = self._effective_entry_price(trade.direction, trade.entry_price)
        effective_exit = self._effective_exit_price(trade.direction, requested_exit_price)
        gross = (effective_exit - effective_entry) * trade.quantity
        if trade.direction == SignalDirection.SELL.value:
            gross = -gross
        commission = self.config.commission_per_unit * trade.quantity * 2.0
        net = gross - commission
        risk_unit = abs(trade.entry_price - trade.stop_loss) * trade.quantity
        r_multiple = net / risk_unit if risk_unit > 0 else 0.0
        return TradeRecord(
            strategy_name=trade.strategy_name,
            strategy_variant=trade.strategy_variant,
            symbol=trade.symbol,
            timeframe=trade.timeframe,
            direction=trade.direction,
            signal_timestamp=trade.signal_timestamp,
            entry_timestamp=trade.entry_timestamp,
            entry_price=effective_entry,
            stop_loss=trade.stop_loss,
            target=trade.target,
            planned_risk_reward=trade.planned_risk_reward,
            quantity=trade.quantity,
            exit_timestamp=bar.timestamp,
            exit_price=effective_exit,
            outcome=outcome,
            r_multiple=r_multiple,
            pnl_gross=gross,
            commission=commission,
            pnl_net=net,
            bars_held=max(0, bar_index - trade.entry_index),
            exit_reason=reason,
        )

    def _close_ambiguous(
        self,
        trade: "_OpenTrade",
        bar: CanonicalOHLC,
        bar_index: int,
    ) -> TradeRecord:
        return TradeRecord(
            strategy_name=trade.strategy_name,
            strategy_variant=trade.strategy_variant,
            symbol=trade.symbol,
            timeframe=trade.timeframe,
            direction=trade.direction,
            signal_timestamp=trade.signal_timestamp,
            entry_timestamp=trade.entry_timestamp,
            entry_price=self._effective_entry_price(trade.direction, trade.entry_price),
            stop_loss=trade.stop_loss,
            target=trade.target,
            planned_risk_reward=trade.planned_risk_reward,
            quantity=trade.quantity,
            exit_timestamp=bar.timestamp,
            exit_price=float(bar.close),
            outcome=TradeOutcome.AMBIGUOUS_SKIPPED,
            r_multiple=0.0,
            pnl_gross=0.0,
            commission=0.0,
            pnl_net=0.0,
            bars_held=max(0, bar_index - trade.entry_index),
            exit_reason="TP_AND_SL_SKIP_TRADE",
        )

    def _effective_entry_price(self, direction: str, raw_price: float) -> float:
        half = self.config.spread / 2.0
        return raw_price + half if direction == SignalDirection.BUY.value else raw_price - half

    def _effective_exit_price(self, direction: str, raw_price: float) -> float:
        half = self.config.spread / 2.0
        return raw_price - half if direction == SignalDirection.BUY.value else raw_price + half

    @staticmethod
    def _validate_signal(
        signal: StrategySignal,
        bar_timestamp: datetime,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Reject signals whose timing, identity or price geometry is inconsistent."""
        if signal.timestamp != bar_timestamp:
            raise ValueError(
                f"strategy signal timestamp {signal.timestamp!r} does not match decision bar {bar_timestamp!r}"
            )
        if signal.symbol.upper() != symbol.upper():
            raise ValueError(
                f"strategy signal symbol {signal.symbol!r} does not match run symbol {symbol!r}"
            )
        if signal.timeframe != timeframe:
            raise ValueError(
                f"strategy signal timeframe {signal.timeframe!r} does not match run timeframe {timeframe!r}"
            )
        if signal.state != SignalState.SIGNAL:
            return
        assert signal.entry is not None and signal.stop_loss is not None and signal.target is not None
        if signal.direction == SignalDirection.BUY:
            valid_geometry = signal.stop_loss < signal.entry < signal.target
        elif signal.direction == SignalDirection.SELL:
            valid_geometry = signal.target < signal.entry < signal.stop_loss
        else:
            valid_geometry = False
        if not valid_geometry:
            raise ValueError("signal price geometry is invalid for its direction")

    @staticmethod
    def _validate_bars(records: Sequence[CanonicalOHLC]) -> None:
        previous = None
        for index, bar in enumerate(records):
            if previous is not None and bar.timestamp <= previous:
                raise ValueError(f"backtest requires strictly increasing timestamps; invalid index {index}")
            previous = bar.timestamp
            prices = (bar.open, bar.high, bar.low, bar.close)
            if any(not isinstance(value, Decimal) or not value.is_finite() or value <= 0 for value in prices):
                raise ValueError(f"invalid OHLC price at index {index}")
            if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close, bar.high):
                raise ValueError(f"impossible OHLC relationship at index {index}")

    def _build_metadata(
        self,
        strategy: Strategy,
        bars: Sequence[CanonicalOHLC],
        analysis: MarketAnalysisSeries,
        symbol: str,
        timeframe: str,
    ) -> dict[str, object]:
        payload = [
            [
                bar.timestamp.isoformat(),
                str(bar.open),
                str(bar.high),
                str(bar.low),
                str(bar.close),
                str(bar.volume) if bar.volume is not None else None,
            ]
            for bar in bars
        ]
        digest = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return {
            "engine_version": self.VERSION,
            "strategy_name": strategy.name,
            "strategy_variant": strategy.variant,
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_engine_version": analysis.engine_version,
            "bar_count": len(bars),
            "data_sha256": digest,
            "execution_config": self.config.to_dict(),
            "signal_execution_policy": "decision_bar_close_then_monitor_from_next_bar",
            "open_trade_policy": "single_active_trade; signals_while_open_are_ignored",
            "intrabar_policy": self.config.intrabar_policy.value,
            "lookahead_policy": "strategy_context_sliced_to_decision_index",
        }


class _OpenTrade:
    """Mutable internal state; never exposed outside the engine."""

    def __init__(
        self,
        *,
        strategy_name: str,
        strategy_variant: str,
        symbol: str,
        timeframe: str,
        direction: str,
        signal_timestamp: datetime,
        entry_timestamp: datetime,
        entry_price: float,
        stop_loss: float,
        target: float,
        planned_risk_reward: float,
        quantity: float,
        entry_index: int,
    ) -> None:
        self.strategy_name = strategy_name
        self.strategy_variant = strategy_variant
        self.symbol = symbol
        self.timeframe = timeframe
        self.direction = direction
        self.signal_timestamp = signal_timestamp
        self.entry_timestamp = entry_timestamp
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target = target
        self.planned_risk_reward = planned_risk_reward
        self.quantity = quantity
        self.entry_index = entry_index

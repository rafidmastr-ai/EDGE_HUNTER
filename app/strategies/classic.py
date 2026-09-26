"""Classic trend/momentum strategy variant used as a Phase 04 baseline."""

from __future__ import annotations

from app.features.models import MarketAnalysisSnapshot
from app.strategies.models import (
    SignalDirection,
    SignalState,
    StrategyConfig,
    StrategyContext,
    StrategySignal,
)
from app.strategies.utils import adaptive_rr, feature_number, recent_high, recent_low, score_flags, valid_signal_prices


class ClassicV1:
    """EMA alignment + RSI/MACD confirmation + recent-range break.

    This is a transparent baseline variant, not a claim about profitability.
    """

    name = "Classic"
    variant = "Classic_V1"

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def generate(self, context: StrategyContext) -> StrategySignal:
        current = context.current
        bars = context.bars
        if not current.warmup_complete or len(bars) < self.config.swing_lookback + 1:
            return _no_signal(context, SignalState.INSUFFICIENT_DATA, "warmup_or_history_incomplete")

        close = feature_number(current, "price.close")
        ema20 = feature_number(current, "trend.ema_20")
        ema50 = feature_number(current, "trend.ema_50")
        ema200 = feature_number(current, "trend.ema_200")
        rsi = feature_number(current, "momentum.rsi")
        histogram = feature_number(current, "momentum.macd_histogram")
        body_ratio = feature_number(current, "structure.body_ratio")
        prior_high = recent_high(bars[:-1], self.config.swing_lookback)
        prior_low = recent_low(bars[:-1], self.config.swing_lookback)
        if any(value is None for value in (close, ema20, ema50, ema200, rsi, histogram, body_ratio)):
            return _no_signal(context, SignalState.INSUFFICIENT_DATA, "required_features_unavailable")
        assert close is not None and ema20 is not None and ema50 is not None and ema200 is not None
        assert rsi is not None and histogram is not None and body_ratio is not None
        if prior_high is None or prior_low is None:
            return _no_signal(context, SignalState.INSUFFICIENT_DATA, "recent_structure_unavailable")

        bullish = (
            ema20 > ema50 > ema200
            and rsi >= 55.0
            and histogram > 0.0
            and close > prior_high
            and body_ratio >= self.config.minimum_body_ratio
        )
        bearish = (
            ema20 < ema50 < ema200
            and rsi <= 45.0
            and histogram < 0.0
            and close < prior_low
            and body_ratio >= self.config.minimum_body_ratio
        )

        if bullish and bearish:
            return _no_signal(context, SignalState.CONFLICT, "bullish_and_bearish_conditions_overlap")
        if not bullish and not bearish:
            return _no_signal(context, SignalState.NO_SIGNAL, "classic_conditions_not_confirmed")

        direction = SignalDirection.BUY if bullish else SignalDirection.SELL
        stop = prior_low if bullish else prior_high
        risk = close - stop if bullish else stop - close
        strength = (body_ratio - self.config.minimum_body_ratio) / (1.0 - self.config.minimum_body_ratio) if self.config.minimum_body_ratio < 1.0 else 1.0
        rr = adaptive_rr(strength, self.config.min_rr, self.config.max_rr)
        if rr is None:
            return _no_signal(context, SignalState.NO_SIGNAL, "invalid_rr_configuration")
        target = close + risk * rr if bullish else close - risk * rr
        if not valid_signal_prices(direction.value, close, stop, target):
            return _no_signal(context, SignalState.NO_SIGNAL, "invalid_price_geometry")
        _, score_inputs = score_flags(
            ema_alignment=(ema20 > ema50 > ema200) if bullish else (ema20 < ema50 < ema200),
            rsi_confirmation=(rsi >= 55.0) if bullish else (rsi <= 45.0),
            macd_confirmation=histogram > 0.0 if bullish else histogram < 0.0,
            range_break=close > prior_high if bullish else close < prior_low,
            candle_body=body_ratio >= self.config.minimum_body_ratio,
        )
        return StrategySignal(
            timestamp=current.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            direction=direction,
            state=SignalState.SIGNAL,
            entry=close,
            stop_loss=stop,
            target=target,
            risk_reward=rr,
            entry_logic="Enter at the decision bar close after trend, momentum and recent-range confirmation.",
            invalidation="Invalidate when the recent structural extreme used for the stop is broken against the signal.",
            stop_loss_logic="Use the opposite side of the pre-break recent range as the structural stop.",
            target_logic="Use one target at the configured dynamic R:R inside the configured research range.",
            evidence=(
                "EMA 20/50/200 alignment",
                "RSI directional confirmation",
                "MACD histogram confirmation",
                "recent-range breakout",
                "minimum candle-body confirmation",
            ),
            score_inputs=score_inputs,
            strategy_name=self.name,
            variant=self.variant,
            metadata={"rr_policy": "adaptive_strength_clamped_to_config", "phase": 4},
        )


def _no_signal(context: StrategyContext, state: SignalState, reason: str) -> StrategySignal:
    return StrategySignal(
        timestamp=context.current.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        direction=SignalDirection.NO_SIGNAL,
        state=state,
        entry=None,
        stop_loss=None,
        target=None,
        risk_reward=None,
        entry_logic="No entry generated.",
        invalidation="No active setup.",
        stop_loss_logic="No stop-loss generated.",
        target_logic="No target generated.",
        evidence=(reason,),
        score_inputs={},
        strategy_name=ClassicV1.name,
        variant=ClassicV1.variant,
        metadata={"reason": reason, "phase": 4},
    )

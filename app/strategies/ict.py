"""ICT-inspired baseline variant using displacement and fair-value-gap logic."""

from __future__ import annotations

from app.strategies.models import SignalDirection, SignalState, StrategyConfig, StrategyContext, StrategySignal
from app.strategies.utils import adaptive_rr, feature_number, recent_high, recent_low, score_flags, valid_signal_prices


class ICTV1:
    """Three-candle FVG + displacement + directional context baseline.

    Session-time killzones are intentionally not assumed here because timezone
    policy and session calendars belong to later data/research configuration.
    """

    name = "ICT"
    variant = "ICT_V1"

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def generate(self, context: StrategyContext) -> StrategySignal:
        bars = context.bars
        current = context.current
        if not current.warmup_complete or len(bars) < 4:
            return _no_signal(context, SignalState.INSUFFICIENT_DATA, "warmup_or_history_incomplete")

        first, middle, last = bars[-3], bars[-2], bars[-1]
        close = float(last.close)
        body_ratio = feature_number(current, "structure.body_ratio")
        ema20 = feature_number(current, "trend.ema_20")
        if body_ratio is None or ema20 is None:
            return _no_signal(context, SignalState.INSUFFICIENT_DATA, "required_features_unavailable")

        bullish_fvg = float(last.low) > float(first.high)
        bearish_fvg = float(last.high) < float(first.low)
        bullish_displacement = close > float(middle.close) and body_ratio >= self.config.minimum_body_ratio
        bearish_displacement = close < float(middle.close) and body_ratio >= self.config.minimum_body_ratio
        bullish_context = close > ema20
        bearish_context = close < ema20
        bullish = bullish_fvg and bullish_displacement and bullish_context
        bearish = bearish_fvg and bearish_displacement and bearish_context

        if bullish and bearish:
            return _no_signal(context, SignalState.CONFLICT, "both_fvg_directions_triggered")
        if not bullish and not bearish:
            return _no_signal(context, SignalState.NO_SIGNAL, "ict_conditions_not_confirmed")

        direction = SignalDirection.BUY if bullish else SignalDirection.SELL
        stop_anchor = min(float(first.low), float(middle.low), float(last.low)) if bullish else max(float(first.high), float(middle.high), float(last.high))
        risk = close - stop_anchor if bullish else stop_anchor - close
        strength = (body_ratio - self.config.minimum_body_ratio) / (1.0 - self.config.minimum_body_ratio) if self.config.minimum_body_ratio < 1.0 else 1.0
        rr = adaptive_rr(strength, self.config.min_rr, self.config.max_rr)
        if rr is None:
            return _no_signal(context, SignalState.NO_SIGNAL, "invalid_rr_configuration")
        target = close + risk * rr if bullish else close - risk * rr
        if not valid_signal_prices(direction.value, close, stop_anchor, target):
            return _no_signal(context, SignalState.NO_SIGNAL, "invalid_price_geometry")

        _, score_inputs = score_flags(
            fair_value_gap=bullish_fvg if bullish else bearish_fvg,
            displacement=bullish_displacement if bullish else bearish_displacement,
            directional_context=bullish_context if bullish else bearish_context,
            body_confirmation=body_ratio >= self.config.minimum_body_ratio,
        )
        return StrategySignal(
            timestamp=current.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            direction=direction,
            state=SignalState.SIGNAL,
            entry=close,
            stop_loss=stop_anchor,
            target=target,
            risk_reward=rr,
            entry_logic="Enter at the decision-bar close after a three-candle fair-value gap, displacement and directional context.",
            invalidation="Invalidate beyond the local three-candle imbalance structure.",
            stop_loss_logic="Place the stop beyond the local three-candle extreme that invalidates the setup.",
            target_logic="Use one dynamic target constrained to the configured R:R range and available structural room.",
            evidence=(
                "three-candle fair-value gap",
                "displacement confirmation",
                "EMA directional context",
                "candle-body confirmation",
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
        strategy_name=ICTV1.name,
        variant=ICTV1.variant,
        metadata={"reason": reason, "phase": 4},
    )

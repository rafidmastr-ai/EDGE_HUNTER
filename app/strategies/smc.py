"""SMC-inspired baseline variant with displacement and liquidity sweep logic."""

from __future__ import annotations

from app.strategies.models import SignalDirection, SignalState, StrategyConfig, StrategyContext, StrategySignal
from app.strategies.utils import adaptive_rr, feature_number, recent_high, recent_low, score_flags, valid_signal_prices


class SMCV1:
    """Liquidity sweep + displacement + market-structure confirmation.

    The implementation is intentionally explicit and testable; it does not claim
    to reproduce every interpretation of Smart Money Concepts.
    """

    name = "SMC"
    variant = "SMC_V1"

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def generate(self, context: StrategyContext) -> StrategySignal:
        bars = context.bars
        current = context.current
        required = max(self.config.swing_lookback + 2, 4)
        if not current.warmup_complete or len(bars) < required:
            return _no_signal(context, SignalState.INSUFFICIENT_DATA, "warmup_or_history_incomplete")

        last = bars[-1]
        previous = bars[-2]
        prior_window = bars[-(self.config.swing_lookback + 2) : -1]
        local_low = min(float(bar.low) for bar in prior_window)
        local_high = max(float(bar.high) for bar in prior_window)
        close = float(last.close)
        prev_close = float(previous.close)
        range_value = float(last.high) - float(last.low)
        body_ratio = feature_number(current, "structure.body_ratio")
        if range_value <= 0 or body_ratio is None:
            return _no_signal(context, SignalState.INSUFFICIENT_DATA, "candle_structure_unavailable")

        bullish_sweep = float(last.low) < local_low and close > local_low
        bearish_sweep = float(last.high) > local_high and close < local_high
        bullish_displacement = close > prev_close and body_ratio >= self.config.minimum_body_ratio
        bearish_displacement = close < prev_close and body_ratio >= self.config.minimum_body_ratio
        bullish_bos = close > float(previous.high)
        bearish_bos = close < float(previous.low)

        bullish = bullish_sweep and bullish_displacement and bullish_bos
        bearish = bearish_sweep and bearish_displacement and bearish_bos
        if bullish and bearish:
            return _no_signal(context, SignalState.CONFLICT, "both_sides_triggered")
        if not bullish and not bearish:
            return _no_signal(context, SignalState.NO_SIGNAL, "smc_conditions_not_confirmed")

        direction = SignalDirection.BUY if bullish else SignalDirection.SELL
        stop = float(last.low) if bullish else float(last.high)
        risk = close - stop if bullish else stop - close
        strength = (body_ratio - self.config.minimum_body_ratio) / (1.0 - self.config.minimum_body_ratio) if self.config.minimum_body_ratio < 1.0 else 1.0
        rr = adaptive_rr(strength, self.config.min_rr, self.config.max_rr)
        if rr is None:
            return _no_signal(context, SignalState.NO_SIGNAL, "invalid_rr_configuration")
        target = close + risk * rr if bullish else close - risk * rr
        if not valid_signal_prices(direction.value, close, stop, target):
            return _no_signal(context, SignalState.NO_SIGNAL, "invalid_price_geometry")

        _, score_inputs = score_flags(
            liquidity_sweep=bullish_sweep if bullish else bearish_sweep,
            displacement=bullish_displacement if bullish else bearish_displacement,
            structure_break=bullish_bos if bullish else bearish_bos,
            body_confirmation=body_ratio >= self.config.minimum_body_ratio,
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
            entry_logic="Enter at the decision-bar close after a liquidity sweep, displacement and structure break.",
            invalidation="Invalidate if price breaks the sweep extreme against the setup.",
            stop_loss_logic="Place the stop beyond the swept liquidity extreme.",
            target_logic="Use one dynamic target constrained to the configured R:R range and available structural room.",
            evidence=(
                "liquidity sweep",
                "displacement candle",
                "local structure break",
                "body-ratio confirmation",
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
        strategy_name=SMCV1.name,
        variant=SMCV1.variant,
        metadata={"reason": reason, "phase": 4},
    )

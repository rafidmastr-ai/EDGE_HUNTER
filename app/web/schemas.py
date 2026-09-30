"""Pydantic request/response schemas for the Phase 09 web API."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


DEFAULT_SYMBOLS = ("EURUSD", "XAUUSD")
SUPPORTED_SYMBOLS = DEFAULT_SYMBOLS
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._:-]*(?:/[A-Z0-9][A-Z0-9._:-]*)?$")
RISK_PRESETS = (0.25, 0.50, 1.00, 1.50, 2.00)


class AnalyzeRequest(BaseModel):
    """User-controlled analysis inputs exposed by the web UI."""

    # None (field empty) = let the EDGE ML models pick the best available recommendation.
    symbol: str | None = Field(default="XAUUSD")
    risk_percent: float = Field(default=1.0, gt=0.0, le=100.0)
    capital: float | None = Field(default=None, gt=0.0)
    lot_mode: str = Field(default="auto")
    lot_size: float | None = Field(default=0.01, gt=0.0, le=100.0)

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        normalized = str(value).strip().upper().replace(" ", "")
        compact = normalized.replace("/", "")
        if not SYMBOL_PATTERN.fullmatch(normalized) or not (4 <= len(compact) <= 32):
            raise ValueError(f"invalid symbol format: {normalized}")
        return normalized

    @field_validator("lot_mode", mode="before")
    @classmethod
    def validate_lot_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"auto", "manual"}:
            raise ValueError("lot_mode must be 'auto' or 'manual'")
        return normalized

    @field_validator("lot_size", mode="after")
    @classmethod
    def normalize_lot_size(cls, value: float | None) -> float | None:
        return None if value is None else round(float(value), 2)


class MarketCandle(BaseModel):
    """Chart candle serialized for the browser."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class CapitalImpact(BaseModel):
    """Capital-dependent estimates shown only when capital was supplied."""

    target_profit: float
    stop_loss_loss: float
    risk_amount: float
    currency: str = "USD"


class AnalysisResponse(BaseModel):
    """Structured API-ready result for the frontend."""

    status: str
    symbol: str
    timestamp: str
    time_zone: str
    primary_timeframe: str
    analyzed_timeframes: list[str]
    direction: str
    confidence: float
    confidence_label_ar: str
    selected_strategy: str | None
    selected_variant: str | None
    entry: float | None
    target: float | None
    stop_loss: float | None
    risk_reward: float | None
    lot_size: float | None
    reasons: list[str]
    strategy_comparison: list[dict[str, Any]]
    directional_support: dict[str, float]
    scoring_components: dict[str, float]
    chart: list[MarketCandle]
    capital_impact: CapitalImpact | None
    metadata: dict[str, Any]

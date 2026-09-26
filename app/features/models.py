"""Canonical market-analysis objects used by the reusable feature engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


FeatureValue = float | int | str | bool | None


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata describing one deterministic analysis feature."""

    name: str
    group: str
    description: str
    source: str
    lookback_bars: int
    requires_volume: bool = False


@dataclass(frozen=True)
class MarketAnalysisSnapshot:
    """Feature snapshot available at one canonical market timestamp."""

    timestamp: datetime
    symbol: str
    timeframe: str
    values: Mapping[str, FeatureValue]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warmup_complete: bool = False

    def get(self, name: str, default: FeatureValue = None) -> FeatureValue:
        """Return a feature value without mutating the immutable snapshot."""
        return self.values.get(name, default)


@dataclass(frozen=True)
class MarketAnalysisSeries:
    """Ordered feature snapshots plus the definitions used to produce them."""

    symbol: str
    timeframe: str
    snapshots: tuple[MarketAnalysisSnapshot, ...]
    definitions: tuple[FeatureDefinition, ...]
    warmup_bars_required: int
    volume_features_enabled: bool
    engine_version: str = "phase03-v1"

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        """Return timestamps in deterministic analysis order."""
        return tuple(snapshot.timestamp for snapshot in self.snapshots)

    def at(self, timestamp: datetime) -> MarketAnalysisSnapshot | None:
        """Find the exact snapshot at a timestamp."""
        for snapshot in self.snapshots:
            if snapshot.timestamp == timestamp:
                return snapshot
        return None

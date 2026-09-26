"""Look-ahead-safe multi-timeframe feature alignment."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from app.features.models import MarketAnalysisSeries, MarketAnalysisSnapshot


@dataclass(frozen=True)
class MultiTimeframeView:
    """The feature snapshots visible at one base timeframe timestamp."""

    base_timestamp: datetime
    frames: Mapping[str, MarketAnalysisSnapshot | None]

    def get(self, timeframe: str) -> MarketAnalysisSnapshot | None:
        """Return the aligned snapshot for a requested timeframe."""
        return self.frames.get(timeframe)


class MultiTimeframeAligner:
    """Align higher timeframe analysis to base timestamps with backward as-of matching."""

    def __init__(self, *, strict_before: bool = False) -> None:
        self.strict_before = strict_before

    def align(
        self,
        base_series: MarketAnalysisSeries,
        other_timeframes: Mapping[str, MarketAnalysisSeries],
    ) -> tuple[MultiTimeframeView, ...]:
        indexes = {
            timeframe: (series.timestamps, series.snapshots)
            for timeframe, series in other_timeframes.items()
        }
        views: list[MultiTimeframeView] = []
        for base_snapshot in base_series.snapshots:
            frames: dict[str, MarketAnalysisSnapshot | None] = {}
            for timeframe, (timestamps, snapshots) in indexes.items():
                position = self._position(timestamps, base_snapshot.timestamp)
                frames[timeframe] = snapshots[position] if position is not None else None
            views.append(MultiTimeframeView(base_snapshot.timestamp, frames))
        return tuple(views)

    def latest_at(
        self,
        series: MarketAnalysisSeries,
        timestamp: datetime,
    ) -> MarketAnalysisSnapshot | None:
        """Return the newest snapshot that is not in the future relative to timestamp."""
        position = self._position(series.timestamps, timestamp)
        return series.snapshots[position] if position is not None else None

    def _position(self, timestamps: Sequence[datetime], timestamp: datetime) -> int | None:
        if not timestamps:
            return None
        index = bisect_left(timestamps, timestamp) - 1 if self.strict_before else bisect_right(timestamps, timestamp) - 1
        return index if index >= 0 else None


"""CSV -> canonical OHLC -> validation pipeline."""

from __future__ import annotations

from pathlib import Path

from app.data.cleaning import drop_synthetic_flat_runs
from app.data.loaders.csv_loader import CSVLoader
from app.data.normalizers.ohlc_normalizer import normalize_row
from app.data.schema import CanonicalOHLC
from app.data.validators.ohlc_validator import DataQualityReport, OHLCValidator


class HistoricalDataPipeline:
    """Deterministic historical data ingestion pipeline."""

    def __init__(self) -> None:
        self.loader = CSVLoader()
        self.validator = OHLCValidator()

    def load(
        self,
        path: Path,
        symbol: str,
        timeframe: str,
    ) -> tuple[list[CanonicalOHLC], DataQualityReport]:
        bars = [normalize_row(row) for row in self.loader.read_rows(path)]

        # Deterministic order before validation.
        bars.sort(key=lambda bar: bar.timestamp)

        # Deterministic deduplication: retain the first record for a timestamp.
        unique: dict = {}
        for bar in bars:
            unique.setdefault(bar.timestamp, bar)

        # Drop synthetic closed-market filler (flat candles repeating the last close).
        cleaned, synthetic = drop_synthetic_flat_runs(list(unique.values()))
        report = self.validator.validate(cleaned, symbol, timeframe)
        report.synthetic_flat_bars_removed = synthetic
        return cleaned, report

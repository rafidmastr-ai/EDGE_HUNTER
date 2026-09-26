"""CLI for running the Phase 06 research matrix over CSV OHLC data."""

from __future__ import annotations

import argparse
import sys

import csv
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.normalizers.ohlc_normalizer import normalize_row
from app.data.schema import CanonicalOHLC
from app.research import ResearchMatrix, ResearchRunner, save_markdown_report, save_research_run


def _normalize_row_with_epoch_compatibility(row: dict[str, str]) -> CanonicalOHLC:
    """Reuse the Phase 02 normalizer and add compatibility for epoch timestamps."""
    timestamp_value = None
    for key in row:
        if key.strip().lower() in {"timestamp", "time", "datetime", "date"}:
            timestamp_value = row[key].strip()
            break
    if timestamp_value:
        try:
            numeric = int(timestamp_value)
        except ValueError:
            numeric = None
        if numeric is not None and abs(numeric) >= 10**11:
            row = dict(row)
            row["timestamp"] = datetime.fromtimestamp(numeric / 1000.0, tz=timezone.utc).isoformat()
    return normalize_row(row)


def load_csv(path: Path) -> tuple[str, tuple[CanonicalOHLC, ...]]:
    symbol = path.stem.upper()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        bars = tuple(_normalize_row_with_epoch_compatibility(dict(row)) for row in reader)
    bars = tuple(sorted(bars, key=lambda item: item.timestamp))
    unique: dict[datetime, CanonicalOHLC] = {}
    for bar in bars:
        unique.setdefault(bar.timestamp, bar)
    return symbol, tuple(unique.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDGE HUNTER Phase 06 research.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/research/phase06"))
    parser.add_argument("--quick", action="store_true", help="Use a reduced matrix for smoke testing.")
    args = parser.parse_args()

    datasets: dict[str, tuple[CanonicalOHLC, ...]] = {}
    sources: dict[str, str] = {}
    for path in sorted(args.data_dir.glob("*.csv")):
        symbol, bars = load_csv(path)
        if bars:
            datasets[symbol] = bars
            sources[symbol] = str(path)

    matrix = ResearchMatrix.quick() if args.quick else ResearchMatrix.default()
    runner = ResearchRunner()
    run = runner.run_dataset(datasets, matrix, sources=sources)
    outputs = save_research_run(run, args.output_dir)
    outputs["report"] = save_markdown_report(run, args.output_dir / "research_report.md")

    print("=" * 64)
    print("        EDGE HUNTER - PHASE 06 RESEARCH")
    print("=" * 64)
    print(f"Run ID: {run.run_id}")
    print(f"Experiments executed: {len(run.experiments)}")
    print(f"Phase 07 candidates: {len(run.candidates)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

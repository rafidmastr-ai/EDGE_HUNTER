
"""CSV discovery and loading for Phase 02."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


class CSVLoader:
    """Discover and load CSV rows without applying market assumptions."""

    def discover(self, root: Path, symbol: str | None = None) -> list[Path]:
        pattern = f"{symbol}*.csv" if symbol else "*.csv"
        return sorted(root.glob(pattern))

    def read_rows(self, path: Path) -> Iterable[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)

"""Export helpers for reproducible backtest results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.backtest.models import BacktestResult


def save_json(result: BacktestResult, path: str | Path) -> Path:
    """Save a backtest result as UTF-8 JSON and return its resolved path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def to_json(result: BacktestResult) -> str:
    """Return a stable JSON representation for tests and integrations."""
    return json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False)


__all__ = ["save_json", "to_json"]


"""Serialization helper for data-quality reports."""

from __future__ import annotations

from dataclasses import asdict

from app.data.validators.ohlc_validator import DataQualityReport


def report_to_dict(report: DataQualityReport) -> dict:
    return asdict(report)

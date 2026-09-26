"""Phase 06 backtest research and strategy-comparison package."""

from app.research.models import (
    MetricSummary,
    ResearchExperiment,
    ResearchMatrix,
    ResearchRun,
    ResearchScreeningConfig,
    ResearchVariant,
    SignalStats,
    default_variants,
)
from app.research.persistence import save_research_run
from app.research.report import build_markdown_report, save_markdown_report
from app.research.runner import ResearchRunner
from app.research.resampling import resample_ohlc, timeframe_seconds

__all__ = [
    "MetricSummary",
    "ResearchExperiment",
    "ResearchMatrix",
    "ResearchRun",
    "ResearchRunner",
    "ResearchScreeningConfig",
    "ResearchVariant",
    "SignalStats",
    "build_markdown_report",
    "default_variants",
    "resample_ohlc",
    "save_markdown_report",
    "save_research_run",
    "timeframe_seconds",
]

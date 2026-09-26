"""CLI smoke runner for Phase 07 using synthetic data or an in-memory dataset."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.schema import CanonicalOHLC
from app.optimization.models import OptimizationCandidate, OptimizationConfig
from app.optimization.report import save_markdown_report
from app.optimization.runner import OptimizationRunner


def make_demo_bars(count: int = 240) -> tuple[CanonicalOHLC, ...]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    bars: list[CanonicalOHLC] = []
    for index in range(count):
        move = 0.20 if (index // 12) % 2 == 0 else -0.15
        open_ = price
        close = price + move
        high = max(open_, close) + 0.05
        low = min(open_, close) - 0.05
        bars.append(
            CanonicalOHLC(
                timestamp=base + timedelta(minutes=index),
                open=Decimal(str(round(open_, 5))),
                high=Decimal(str(round(high, 5))),
                low=Decimal(str(round(low, 5))),
                close=Decimal(str(round(close, 5))),
            )
        )
        price = close
    return tuple(bars)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDGE HUNTER Phase 07 smoke optimization.")
    parser.add_argument("--output-dir", default="reports/optimization/phase07")
    args = parser.parse_args()

    candidate = OptimizationCandidate(
        strategy_name="Classic",
        base_config={
            "min_rr": 1.5,
            "max_rr": 2.0,
            "swing_lookback": 5,
            "minimum_body_ratio": 0.55,
            "minimum_confirmation_score": 2.0,
        },
        research_reason="Phase 06 tested R:R, body-ratio and swing-lookback dimensions.",
    )
    runner = OptimizationRunner(
        optimization_config=OptimizationConfig(
            min_validation_trades=0,
            min_oos_trades=0,
            top_train_pool_size=4,
            candidate_pool_size=2,
            walk_forward_folds=2,
        )
    )
    result = runner.optimize(
        candidate,
        make_demo_bars(),
        symbol="XAUUSD",
        timeframe="M1",
        space=runner.default_space(),
        output_dir=args.output_dir,
    )
    save_markdown_report(result, Path(args.output_dir) / "optimization_report.md")
    print(f"Phase 07 run: {result.run_id}")
    print(f"Evaluations: {len(result.evaluations)}")
    print(f"Validated candidates: {len(result.final_candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the Classic / SMC / ICT strategies with machine learning.

Usage (from the project root):

    python scripts/train_strategies.py

What it does:
1. Labels the live setups recorded by the web app (after each analysis) using the
   candles that followed them, fetched from the live provider (Twelve Data).
2. Replays every strategy over each historical CSV in ``data/raw`` (file name =
   symbol, e.g. ``BTCUSD.csv``) on M5 / M15 / H1 and labels every setup
   (target before stop-loss or not).
3. Trains one model per strategy on historical + live setups and keeps it only if
   it improves out-of-sample results.
4. Saves the models; the running web app picks them up automatically.

Options:
    --data-dir PATH                 folder with historical CSV files (default: data/raw)
    --max-bars-per-timeframe N      use only the most recent N bars per timeframe (faster)
    --horizon-bars N                candles allowed for a setup to reach TP/SL (default: settings)
    --skip-live                     do not fetch candles to resolve recorded live setups
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.database import Database  # noqa: E402
from app.db.migrations import MigrationRunner  # noqa: E402
from app.learning.strategy_learning import format_report, run_training  # noqa: E402
from app.providers.factory import build_live_provider  # noqa: E402
from config.config_hunter import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train EDGE HUNTER strategies with machine learning")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--max-bars-per-timeframe", type=int, default=None)
    parser.add_argument("--horizon-bars", type=int, default=None)
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    data_dir = args.data_dir if args.data_dir.is_absolute() else (PROJECT_ROOT / args.data_dir)
    if not any(data_dir.glob("*.csv")):
        print(f"No CSV files found in {data_dir}")
        return 1

    database = Database(settings.database_path)
    try:
        MigrationRunner(database).apply_all()
        provider = None
        if not args.skip_live:
            candidate = build_live_provider(settings)
            if getattr(candidate, "configured", False):
                provider = candidate
        bundle = run_training(
            data_dir=data_dir,
            database=database,
            model_path=settings.strategy_ml_model_path,
            provider=provider,
            horizon_bars=args.horizon_bars or settings.strategy_ml_horizon_bars,
            max_bars_per_timeframe=args.max_bars_per_timeframe,
            progress=lambda message: print(message, flush=True),
        )
    finally:
        database.close()

    report = format_report(bundle)
    report_path = settings.strategy_ml_model_path.with_name("strategy_learning_report.txt")
    report_path.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)
    print(f"\nReport saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

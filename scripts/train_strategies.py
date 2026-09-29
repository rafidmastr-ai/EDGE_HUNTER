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
    --symbols XAUUSD,EURUSD         train only on these symbols
    --timeframes M5,M15,H1          timeframes to replay (default: M5,M15,H1)
    --test-fraction 0.2             test on the last share of each symbol's bars before --test-end
    --test-end 2025-01-01           end (exclusive, UTC) of the test period
    --cost XAUUSD=0.20              round-trip cost per trade in price units (repeatable)
    --out DIR                       write model + report to DIR instead of the live model path

Example (XAUUSD, test = 20% ending 2024-12-31, not applied to the live app):

    python scripts/train_strategies.py --skip-live --symbols XAUUSD --test-fraction 0.2 \
        --test-end 2025-01-01 --cost XAUUSD=0.20 --out data/models/xauusd_split
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.database import Database  # noqa: E402
from app.db.migrations import MigrationRunner  # noqa: E402
from app.data.mt5_history import is_mt5_symbol_dir  # noqa: E402
from app.learning.strategy_learning import ANALYSIS_TIMEFRAMES, format_report, run_training  # noqa: E402
from app.providers.factory import build_live_provider  # noqa: E402
from config.config_hunter import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train EDGE HUNTER strategies with machine learning")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--max-bars-per-timeframe", type=int, default=None)
    parser.add_argument("--horizon-bars", type=int, default=None)
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--timeframes", default=",".join(ANALYSIS_TIMEFRAMES))
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--cost", action="append", default=[], metavar="SYMBOL=PRICE")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if (args.test_fraction is None) != (args.test_end is None):
        parser.error("--test-fraction and --test-end go together")
    if args.test_fraction is not None and not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be in (0, 1)")
    test_end = datetime.fromisoformat(args.test_end).replace(tzinfo=timezone.utc) if args.test_end else None
    cost: dict[str, float] = {}
    for item in args.cost:
        symbol, _, value = item.partition("=")
        cost[symbol.strip().upper()] = float(value)
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()] if args.symbols else None
    timeframes = tuple(item.strip().upper() for item in args.timeframes.split(",") if item.strip())

    settings = load_settings()
    data_dir = args.data_dir if args.data_dir.is_absolute() else (PROJECT_ROOT / args.data_dir)
    if not any(data_dir.glob("*.csv")) and not any(is_mt5_symbol_dir(path) for path in data_dir.iterdir()):
        print(f"No CSV files or MT5 symbol folders found in {data_dir}")
        return 1
    model_path = settings.strategy_ml_model_path
    if args.out is not None:
        out_dir = args.out if args.out.is_absolute() else (PROJECT_ROOT / args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / model_path.name

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
            model_path=model_path,
            provider=provider,
            horizon_bars=args.horizon_bars or settings.strategy_ml_horizon_bars,
            max_bars_per_timeframe=args.max_bars_per_timeframe,
            progress=lambda message: print(message, flush=True),
            symbols=symbols,
            timeframes=timeframes,
            test_fraction=args.test_fraction,
            test_end=test_end,
            cost_per_trade=cost,
        )
    finally:
        database.close()

    report = format_report(bundle)
    report_path = model_path.with_name("strategy_learning_report.txt")
    report_path.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)
    print(f"\nReport saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a consistent online SQLite backup and prune old backups."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_hunter import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="EDGE HUNTER SQLite backup")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--retention-days", type=int, default=14)
    args = parser.parse_args()

    settings = load_settings()
    source = settings.database_path
    if not source.exists():
        raise SystemExit(f"Database not found: {source}")

    output_dir = args.output_dir or (source.parent / "backups")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir / f"edge_hunter_{stamp}.db"

    src = sqlite3.connect(source)
    dst = sqlite3.connect(destination)
    try:
        src.execute("PRAGMA busy_timeout=30000")
        src.backup(dst)
        check = dst.execute("PRAGMA integrity_check").fetchone()
        if not check or check[0] != "ok":
            raise RuntimeError("backup integrity check failed")
    finally:
        dst.close()
        src.close()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, args.retention_days))
    for item in output_dir.glob("edge_hunter_*.db"):
        if item == destination:
            continue
        if datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc) < cutoff:
            item.unlink(missing_ok=True)

    print(f"BACKUP PASS: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

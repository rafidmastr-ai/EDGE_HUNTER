"""Restore an SQLite backup after the application has been stopped."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_hunter import load_settings


def check_integrity(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
    finally:
        conn.close()
    if not result or result[0] != "ok":
        raise RuntimeError(f"source backup failed integrity check: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="EDGE HUNTER SQLite restore")
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()

    settings = load_settings()
    target = settings.database_path
    source = args.backup.resolve()
    if not source.exists():
        raise SystemExit(f"Backup not found: {source}")
    check_integrity(source)

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(target, target.with_name(f"{target.stem}.pre_restore_{stamp}{target.suffix}"))

    shutil.copy2(source, target)
    check_integrity(target)
    print(f"RESTORE PASS: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

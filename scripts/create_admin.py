"""Create or update a local EDGE HUNTER admin account without hard-coded credentials."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

# Running a script as ``python scripts/create_admin.py`` puts ``scripts/``
# first on sys.path.  Ensure the project root is first as well, and explicitly
# select this project's ``app`` package rather than an unrelated installed
# package that may also be named ``app``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.auth.security import hash_password, normalize_email, validate_password
from app.db.database import Database
from app.db.migrations import MigrationRunner
from config.config_hunter import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an EDGE HUNTER admin user")
    parser.add_argument("--email", required=True, help="admin email address")
    args = parser.parse_args()

    try:
        email = normalize_email(args.email)
        password = validate_password(getpass.getpass("Admin password: "))
        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            raise ValueError("passwords do not match")
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    settings = load_settings()
    database = Database(settings.database_path)
    try:
        MigrationRunner(database).apply_all()
        existing = database.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        hashed = hash_password(password, settings.password_iterations)
        if existing:
            database.execute(
                "UPDATE users SET password_hash = ?, role = 'admin', status = 'active' WHERE id = ?",
                (hashed, existing["id"]),
            )
        else:
            # Admins have no trial requirement and begin with a neutral timestamp.
            from app.auth.repository import utc_now

            now = utc_now().isoformat()
            database.execute(
                """
                INSERT INTO users(email, password_hash, role, status, trial_started_at,
                                  trial_expires_at, created_at)
                VALUES (?, ?, 'admin', 'active', ?, ?, ?)
                """,
                (email, hashed, now, now, now),
            )
        database.commit()
        print(f"ADMIN READY: {email}")
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())

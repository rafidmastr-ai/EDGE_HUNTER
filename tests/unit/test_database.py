import tempfile
import unittest
from pathlib import Path

from app.db.database import Database
from app.db.migrations import MigrationRunner


class DatabaseTests(unittest.TestCase):
    def test_phase10_tables_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "auth.db")
            try:
                MigrationRunner(db).apply_all()
                names = {row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
                self.assertTrue({"users", "trial_devices", "subscriptions", "subscription_codes", "auth_sessions", "audit_logs"}.issubset(names))
            finally:
                db.close()

    def test_migrations_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "test.db")
            runner = MigrationRunner(db)
            try:
                runner.apply_all()
                runner.apply_all()

                rows = db.execute(
                    "SELECT version, name FROM schema_migrations ORDER BY version"
                ).fetchall()

                self.assertEqual(len(rows), 12)
                self.assertEqual(rows[0]["version"], 1)
                self.assertEqual(rows[0]["name"], "foundation")
                self.assertEqual(rows[1]["version"], 2)
                self.assertEqual(rows[1]["name"], "auth_subscriptions_admin")
                self.assertEqual(rows[2]["version"], 3)
                self.assertEqual(rows[2]["name"], "learning_foundation")
                self.assertEqual(rows[3]["version"], 4)
                self.assertEqual(rows[3]["name"], "feature_snapshot_persistence")
                self.assertEqual(rows[4]["version"], 5)
                self.assertEqual(rows[4]["name"], "learning_outcome_label_version")
            finally:
                # Windows cannot remove the temporary SQLite file while the
                # connection is still open. Always release it before cleanup.
                db.close()


if __name__ == "__main__":
    unittest.main()

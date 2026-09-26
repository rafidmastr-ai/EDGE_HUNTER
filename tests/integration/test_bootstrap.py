import os
import tempfile
import unittest
from pathlib import Path

from app.core.bootstrap import create_application


class BootstrapTests(unittest.TestCase):
    def test_core_components_import_and_database_can_start(self):
        # Uses an isolated database path for the integration check.
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("EDGE_HUNTER_DATABASE_PATH")
            application = None
            os.environ["EDGE_HUNTER_DATABASE_PATH"] = str(Path(tmp) / "edge.db")
            try:
                application = create_application()
                application.start()

                row = application.database.execute(
                    "SELECT COUNT(*) AS count FROM schema_migrations"
                ).fetchone()
                self.assertEqual(row["count"], 11)
            finally:
                # Release SQLite before TemporaryDirectory removes the file.
                if application is not None:
                    application.database.close()
                if old is None:
                    os.environ.pop("EDGE_HUNTER_DATABASE_PATH", None)
                else:
                    os.environ["EDGE_HUNTER_DATABASE_PATH"] = old


if __name__ == "__main__":
    unittest.main()

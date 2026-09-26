from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.feature_store import FeatureSnapshot, FeatureStore


class FeatureStoreIntegrationTests(unittest.TestCase):
    def test_feature_snapshot_survives_database_reload(self):
        timestamp = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edge.db"
            first_db = Database(path)
            MigrationRunner(first_db).apply_all()
            snapshot = FeatureSnapshot(
                timestamp=timestamp,
                symbol="XAUUSD",
                timeframe="M15",
                feature_engine_version="phase03-v1",
                feature_schema_version="phase03-feature-schema-v1",
                features={"price.close": 2500.0, "momentum.rsi": 58.25},
                created_at=timestamp,
            )
            FeatureStore(first_db).save(snapshot)
            first_db.close()

            second_db = Database(path)
            try:
                MigrationRunner(second_db).apply_all()
                restored = FeatureStore(second_db).load(snapshot.snapshot_id)
                self.assertIsNotNone(restored)
                self.assertEqual(restored.to_dict(), snapshot.to_dict())
            finally:
                second_db.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sqlite3

from app.web.admin_code_diagnostics import save_admin_code_generation_diagnostic


class AdminCodeDiagnosticsTests(unittest.TestCase):
    def test_failure_is_saved_automatically_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "edge.db"
            connection = sqlite3.connect(db_path)
            connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, role TEXT, status TEXT, trial_expires_at TEXT, created_at TEXT, last_login_at TEXT)")
            connection.execute("CREATE TABLE subscription_codes (id INTEGER PRIMARY KEY, code_hint TEXT, duration_months INTEGER, status TEXT, assigned_user_id INTEGER, expires_at TEXT, created_at TEXT, redeemed_at TEXT)")
            connection.execute("CREATE TABLE audit_logs (id INTEGER PRIMARY KEY, actor_user_id INTEGER, action TEXT, target_type TEXT, target_id TEXT, details_json TEXT, created_at TEXT)")
            connection.execute("CREATE TABLE subscriptions (id INTEGER PRIMARY KEY, user_id INTEGER, starts_at TEXT, expires_at TEXT, status TEXT, source_code_id INTEGER, created_at TEXT)")
            connection.execute("INSERT INTO users VALUES (2, 'admin@example.com', 'admin', 'active', '2030-01-01', '2026-01-01', '2026-09-26')")
            connection.execute("INSERT INTO subscription_codes VALUES (1, 'DDLD', 1, 'available', NULL, NULL, '2026-09-26', NULL)")
            connection.commit()
            connection.close()

            database = SimpleNamespace(path=db_path, connection=sqlite3.connect(db_path))
            database.connection.row_factory = sqlite3.Row
            database.connection.execute("PRAGMA foreign_keys=ON")
            settings = SimpleNamespace(environment='development', data_mode='live', live_provider_name='twelvedata', log_dir=root / 'logs')
            request = SimpleNamespace(method='POST', url=SimpleNamespace(path='/api/admin/codes'), client=SimpleNamespace(host='127.0.0.1'), state=SimpleNamespace(request_id='req-123'), headers={'X-CSRF-Token': 'present'})
            payload = SimpleNamespace(duration_months=1, quantity=1)
            exc = sqlite3.IntegrityError('FOREIGN KEY constraint failed')
            exc.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY
            exc.sqlite_errorname = 'SQLITE_CONSTRAINT_FOREIGNKEY'

            path = save_admin_code_generation_diagnostic(database=database, settings=settings, request=request, admin_user_id=2, payload=payload, exc=exc)
            self.assertTrue(path.exists())
            latest = root / 'logs' / 'admin_code_generation_latest.json'
            self.assertTrue(latest.exists())
            data = json.loads(latest.read_text(encoding='utf-8'))
            self.assertEqual(data['operation']['admin_user_id'], 2)
            self.assertEqual(data['exception']['sqlite_errorname'], 'SQLITE_CONSTRAINT_FOREIGNKEY')
            self.assertNotIn('X-CSRF-Token', latest.read_text(encoding='utf-8'))
            database.connection.close()


if __name__ == '__main__':
    unittest.main()

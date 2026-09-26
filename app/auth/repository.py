"""SQLite repository for accounts, sessions, subscriptions and audit data."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.auth.models import AuthUser, SessionContext
from app.auth.security import token_hash
from app.db.database import Database


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class AuthRepository:
    """Data access layer; callers do not construct raw SQL tables themselves."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_user(
        self,
        *,
        email: str,
        password_hash: str,
        trial_started_at: datetime,
        trial_expires_at: datetime,
        device_hash: str,
    ) -> int:
        cursor = self.database.execute(
            """
            INSERT INTO users(
                email, password_hash, role, status,
                trial_started_at, trial_expires_at,
                device_hash, created_at
            ) VALUES (?, ?, 'user', 'active', ?, ?, ?, ?)
            """,
            (
                email,
                password_hash,
                trial_started_at.isoformat(),
                trial_expires_at.isoformat(),
                device_hash,
                trial_started_at.isoformat(),
            ),
        )
        self.database.commit()
        return int(cursor.lastrowid)

    def user_by_email(self, email: str) -> sqlite3.Row | None:
        return self.database.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    def user_by_id(self, user_id: int) -> sqlite3.Row | None:
        return self.database.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    def create_session(self, user_id: int, *, scope: str, raw_token: str, csrf_token: str, expires_at: datetime) -> None:
        now = utc_now().isoformat()
        self.database.execute(
            """
            INSERT INTO auth_sessions(token_hash, user_id, scope, csrf_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (token_hash(raw_token), user_id, scope, token_hash(csrf_token), now, expires_at.isoformat()),
        )
        self.database.commit()

    def session_row(self, raw_token: str) -> sqlite3.Row | None:
        return self.database.execute(
            """
            SELECT s.*, u.email, u.role, u.status,
                   u.trial_started_at, u.trial_expires_at, u.created_at,
                   u.last_login_at, u.device_hash
            FROM auth_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_hash(raw_token),),
        ).fetchone()

    def delete_session(self, raw_token: str) -> None:
        self.database.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash(raw_token),))
        self.database.commit()

    def delete_user_sessions(self, user_id: int) -> None:
        self.database.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))
        self.database.commit()

    def rebind_session_user(self, session_id: int, user_id: int) -> None:
        """Repair a stale session identity after an account was recreated."""
        self.database.execute(
            "UPDATE auth_sessions SET user_id = ? WHERE id = ?",
            (user_id, session_id),
        )
        self.database.commit()

    def update_last_login(self, user_id: int, when: datetime) -> None:
        self.database.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (when.isoformat(), user_id))
        self.database.commit()

    def bind_device(self, user_id: int, device_hash: str) -> None:
        self.database.execute("UPDATE users SET device_hash = ? WHERE id = ?", (device_hash, user_id))
        self.database.commit()

    def clear_device(self, user_id: int) -> None:
        self.database.execute("UPDATE users SET device_hash = NULL WHERE id = ?", (user_id,))
        self.database.commit()

    def add_trial_device(self, device_hash: str, user_id: int, when: datetime) -> bool:
        try:
            self.database.execute(
                "INSERT INTO trial_devices(device_hash, user_id, first_trial_at) VALUES (?, ?, ?)",
                (device_hash, user_id, when.isoformat()),
            )
            self.database.commit()
            return True
        except Exception:
            self.database.connection.rollback()
            return False

    def trial_device_exists(self, device_hash: str) -> bool:
        row = self.database.execute("SELECT 1 FROM trial_devices WHERE device_hash = ?", (device_hash,)).fetchone()
        return row is not None

    def active_subscription(self, user_id: int, now: datetime | None = None) -> sqlite3.Row | None:
        now = now or utc_now()
        row = self.database.execute(
            """
            SELECT * FROM subscriptions
            WHERE user_id = ? AND status = 'active' AND expires_at > ?
            ORDER BY expires_at DESC LIMIT 1
            """,
            (user_id, now.isoformat()),
        ).fetchone()
        return row

    def create_subscription(self, user_id: int, starts_at: datetime, expires_at: datetime, source_code_id: int | None) -> int:
        cursor = self.database.execute(
            """
            INSERT INTO subscriptions(user_id, starts_at, expires_at, status, source_code_id, created_at)
            VALUES (?, ?, ?, 'active', ?, ?)
            """,
            (user_id, starts_at.isoformat(), expires_at.isoformat(), source_code_id, utc_now().isoformat()),
        )
        self.database.commit()
        return int(cursor.lastrowid)

    def create_code(self, code_hash_value: str, hint: str, duration_months: int, expires_at: datetime | None) -> int:
        cursor = self.database.execute(
            """
            INSERT INTO subscription_codes(
                code_hash, code_hint, duration_months, status, expires_at, created_at
            ) VALUES (?, ?, ?, 'available', ?, ?)
            """,
            (code_hash_value, hint, duration_months, expires_at.isoformat() if expires_at else None, utc_now().isoformat()),
        )
        return int(cursor.lastrowid)

    def code_by_hash(self, code_hash_value: str) -> sqlite3.Row | None:
        return self.database.execute("SELECT * FROM subscription_codes WHERE code_hash = ?", (code_hash_value,)).fetchone()

    def mark_code_redeemed(self, code_id: int, user_id: int, when: datetime) -> None:
        self.database.execute(
            """
            UPDATE subscription_codes
            SET status = 'redeemed', assigned_user_id = ?, redeemed_at = ?
            WHERE id = ?
            """,
            (user_id, when.isoformat(), code_id),
        )

    def revoke_subscription(self, subscription_id: int, reason: str | None = None) -> bool:
        cursor = self.database.execute(
            """
            UPDATE subscriptions
            SET status = 'revoked', revoked_at = ?, revoke_reason = ?
            WHERE id = ? AND status = 'active'
            """,
            (utc_now().isoformat(), reason, subscription_id),
        )
        self.database.commit()
        return cursor.rowcount == 1

    def extend_subscription(self, subscription_id: int, starts_at: datetime, expires_at: datetime) -> bool:
        cursor = self.database.execute(
            """
            UPDATE subscriptions
            SET starts_at = ?, expires_at = ?, status = 'active', revoked_at = NULL, revoke_reason = NULL
            WHERE id = ?
            """,
            (starts_at.isoformat(), expires_at.isoformat(), subscription_id),
        )
        self.database.commit()
        return cursor.rowcount == 1

    def subscriptions_for_user(self, user_id: int) -> list[sqlite3.Row]:
        return self.database.execute(
            "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY expires_at DESC", (user_id,)
        ).fetchall()

    def add_audit(self, actor_user_id: int | None, action: str, target_type: str, target_id: str | None, details: dict[str, Any]) -> None:
        self.database.execute(
            """
            INSERT INTO audit_logs(actor_user_id, action, target_type, target_id, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (actor_user_id, action, target_type, target_id, json.dumps(details, ensure_ascii=False, sort_keys=True), utc_now().isoformat()),
        )
        self.database.commit()

    def list_users(self) -> list[sqlite3.Row]:
        return self.database.execute(
            "SELECT id, email, role, status, trial_started_at, trial_expires_at, created_at, last_login_at, device_hash FROM users ORDER BY created_at DESC"
        ).fetchall()

    def list_codes(self) -> list[sqlite3.Row]:
        return self.database.execute(
            """
            SELECT c.*, u.email AS assigned_email
            FROM subscription_codes c
            LEFT JOIN users u ON u.id = c.assigned_user_id
            ORDER BY c.created_at DESC
            """
        ).fetchall()

    def list_audit(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.database.execute(
            """
            SELECT a.*, u.email AS actor_email
            FROM audit_logs a
            LEFT JOIN users u ON u.id = a.actor_user_id
            ORDER BY a.id DESC LIMIT ?
            """,
            (max(1, min(limit, 500)),),
        ).fetchall()

    def dashboard_counts(self, now: datetime | None = None) -> dict[str, int]:
        now = now or utc_now()
        users = self.list_users()
        trial = 0
        subscribed = 0
        expired = 0
        for row in users:
            if row["role"] == "admin":
                continue
            if dt(row["trial_expires_at"]) and dt(row["trial_expires_at"]) > now and not self.active_subscription(int(row["id"]), now):
                trial += 1
            elif self.active_subscription(int(row["id"]), now):
                subscribed += 1
            else:
                expired += 1
        codes = self.list_codes()
        return {
            "users": len([row for row in users if row["role"] != "admin"]),
            "trial_active": trial,
            "subscription_active": subscribed,
            "access_expired": expired,
            "codes_available": sum(1 for row in codes if row["status"] == "available" and (row["expires_at"] is None or dt(row["expires_at"]) > now)),
            "codes_redeemed": sum(1 for row in codes if row["status"] == "redeemed"),
            "codes_revoked": sum(1 for row in codes if row["status"] == "revoked"),
        }

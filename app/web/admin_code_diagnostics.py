"""Automatic, secret-free diagnostics for admin subscription-code generation failures."""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


_JSONL_NAME = "admin_code_generation_errors.jsonl"
_LATEST_NAME = "admin_code_generation_latest.json"
_WRITE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary>"
    return str(value)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _log_dir(settings: Any) -> Path:
    configured = getattr(settings, "log_dir", None)
    if configured is None:
        return _project_root() / "logs"
    path = Path(configured)
    return path if path.is_absolute() else _project_root() / path


def _schema(connection: sqlite3.Connection, table: str) -> str | None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def _query_one(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    return connection.execute(sql, params).fetchone()


def collect_admin_code_diagnostic(
    *, database: Any, settings: Any, request: Any, admin_user_id: int, payload: Any, exc: BaseException
) -> dict[str, Any]:
    """Collect enough runtime/DB state to diagnose the failed generation safely."""
    connection = database.connection
    db_path = Path(database.path)
    resolved = db_path.resolve()

    event: dict[str, Any] = {
        "diagnostic_version": 3,
        "timestamp_utc": _utc_now(),
        "process": {
            "pid": os.getpid(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "request": {
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "csrf_header_present": bool(request.headers.get("X-CSRF-Token")),
        },
        "operation": {
            "name": "admin_generate_subscription_codes",
            "admin_user_id": int(admin_user_id),
            "duration_months": int(payload.duration_months),
            "quantity": int(payload.quantity),
        },
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
            "sqlite_errorcode": getattr(exc, "sqlite_errorcode", None),
            "sqlite_errorname": getattr(exc, "sqlite_errorname", None),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
        "runtime": {
            "environment": getattr(settings, "environment", None),
            "data_mode": getattr(settings, "data_mode", None),
            "live_provider": getattr(settings, "live_provider_name", None),
            "database_configured_path": str(db_path),
            "diagnostic_log_dir": str(_log_dir(settings).resolve()),
        },
    }

    db: dict[str, Any] = {
        "resolved_path": str(resolved),
        "file_exists": db_path.is_file(),
        "file_size_bytes": db_path.stat().st_size if db_path.is_file() else None,
        "connection_in_transaction": bool(connection.in_transaction),
        "isolation_level": connection.isolation_level,
        "sqlite_version": sqlite3.sqlite_version,
    }

    for pragma in ("foreign_keys", "journal_mode", "synchronous"):
        try:
            db.setdefault("pragma", {})[pragma] = _safe(connection.execute(f"PRAGMA {pragma}").fetchone()[0])
        except Exception as pragma_exc:  # diagnostics must never hide the root error
            db.setdefault("pragma", {})[pragma] = f"<error: {type(pragma_exc).__name__}: {pragma_exc}>"

    try:
        db["database_list"] = [
            {"seq": int(row[0]), "name": str(row[1]), "file": str(row[2])}
            for row in connection.execute("PRAGMA database_list").fetchall()
        ]
    except Exception as db_exc:
        db["database_list"] = [f"<error: {type(db_exc).__name__}: {db_exc}>"]

    try:
        row = _query_one(
            connection,
            "SELECT id, email, role, status, trial_expires_at, created_at, last_login_at FROM users WHERE id = ?",
            (int(admin_user_id),),
        )
        db["session_admin_row"] = dict(row) if row else None
    except Exception as row_exc:
        db["session_admin_row"] = f"<error: {type(row_exc).__name__}: {row_exc}>"

    try:
        db["admin_rows"] = [
            {"id": int(row[0]), "email": str(row[1]), "role": str(row[2]), "status": str(row[3])}
            for row in connection.execute(
                "SELECT id, email, role, status FROM users WHERE role='admin' ORDER BY id"
            ).fetchall()
        ]
    except Exception as row_exc:
        db["admin_rows"] = [f"<error: {type(row_exc).__name__}: {row_exc}>"]

    db["counts"] = {}
    for name, sql in {
        "users": "SELECT COUNT(*) FROM users",
        "admins": "SELECT COUNT(*) FROM users WHERE role='admin'",
        "subscription_codes": "SELECT COUNT(*) FROM subscription_codes",
        "audit_logs": "SELECT COUNT(*) FROM audit_logs",
    }.items():
        try:
            db["counts"][name] = int(connection.execute(sql).fetchone()[0])
        except Exception as count_exc:
            db["counts"][name] = f"<error: {type(count_exc).__name__}: {count_exc}>"

    try:
        row = connection.execute(
            "SELECT id, code_hint, duration_months, status, assigned_user_id, expires_at, created_at, redeemed_at "
            "FROM subscription_codes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        db["latest_code_row"] = dict(row) if row else None
    except Exception as code_exc:
        db["latest_code_row"] = f"<error: {type(code_exc).__name__}: {code_exc}>"

    try:
        db["foreign_key_check"] = [
            {"table": str(row[0]), "rowid": _safe(row[1]), "parent": str(row[2]), "fkid": _safe(row[3])}
            for row in connection.execute("PRAGMA foreign_key_check").fetchall()
        ]
    except Exception as fk_exc:
        db["foreign_key_check"] = [f"<error: {type(fk_exc).__name__}: {fk_exc}>"]

    db["schema"] = {
        table: _schema(connection, table) for table in ("users", "subscription_codes", "audit_logs", "subscriptions")
    }
    event["database"] = db
    return event


def save_admin_code_generation_diagnostic(
    *, database: Any, settings: Any, request: Any, admin_user_id: int, payload: Any, exc: BaseException
) -> Path:
    """Persist a failure automatically; return the JSONL path for logging."""
    directory = _log_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    event = collect_admin_code_diagnostic(
        database=database,
        settings=settings,
        request=request,
        admin_user_id=admin_user_id,
        payload=payload,
        exc=exc,
    )

    jsonl_path = directory / _JSONL_NAME
    latest_path = directory / _LATEST_NAME
    encoded = json.dumps(event, ensure_ascii=False, default=_safe, separators=(",", ":"))
    with _WRITE_LOCK:
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
        with latest_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=_safe, indent=2))
            handle.flush()
    return jsonl_path.resolve()

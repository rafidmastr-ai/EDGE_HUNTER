"""Application service implementing account, trial, session and subscription rules."""

from __future__ import annotations

import calendar
import hmac
import sqlite3
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

from app.auth.models import AccessState, AuthUser, GeneratedCode, SessionContext
from app.auth.repository import AuthRepository, dt, utc_now
from app.auth.security import (
    generate_csrf_token,
    generate_device_token,
    generate_subscription_code,
    hash_password,
    normalize_email,
    normalize_subscription_code,
    token_hash,
    validate_password,
    verify_password,
)
from app.db.database import Database


class AuthError(RuntimeError):
    """Base error mapped to a safe API response."""

    code = "auth_error"
    status_code = 400

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.details = details or {}


class ValidationError(AuthError):
    code = "validation_error"
    status_code = 422


class AuthenticationError(AuthError):
    code = "unauthorized"
    status_code = 401


class ForbiddenError(AuthError):
    code = "forbidden"
    status_code = 403


class SessionExpiredError(AuthenticationError):
    code = "session_expired"


class RateLimitError(AuthError):
    code = "rate_limited"
    status_code = 429


class AuthService:
    """High-level security/access service used by web and admin routes."""

    def __init__(
        self,
        database: Database,
        *,
        session_ttl_hours: int = 24,
        password_iterations: int = 600_000,
    ) -> None:
        self.database = database
        self.repo = AuthRepository(database)
        self.session_ttl = timedelta(hours=max(1, session_ttl_hours))
        self.password_iterations = max(100_000, min(password_iterations, 2_000_000))
        self._write_lock = RLock()

    def register(
        self,
        email: str,
        password: str,
        device_token: str | None = None,
    ) -> tuple[AuthUser, str, str, str]:
        email = self._normalize_email(email)
        password = self._validate_password(password)
        device_token = device_token or generate_device_token()
        device_hash = token_hash(device_token)
        now = utc_now()

        if self.repo.user_by_email(email) is not None:
            raise AuthError("an account with this email already exists", code="email_exists", status_code=409)
        if self.repo.trial_device_exists(device_hash):
            raise ForbiddenError("free trial is already associated with this device", code="trial_already_used")

        trial_expires = now + timedelta(days=7)
        try:
            user_id = self.repo.create_user(
                email=email,
                password_hash=hash_password(password, self.password_iterations),
                trial_started_at=now,
                trial_expires_at=trial_expires,
                device_hash=device_hash,
            )
        except sqlite3.IntegrityError as exc:
            raise AuthError("unable to create account", code="email_exists", status_code=409) from exc

        if not self.repo.add_trial_device(device_hash, user_id, now):
            self.database.execute("DELETE FROM users WHERE id = ?", (user_id,))
            self.database.commit()
            raise ForbiddenError("free trial is already associated with this device", code="trial_already_used")

        user = self.user_from_row(self.repo.user_by_id(user_id))
        session_token, csrf_token = self._create_session(user_id, scope="user")
        self.repo.add_audit(user_id, "register", "user", str(user_id), {"trial_days": 7})
        return user, session_token, csrf_token, device_token

    def login(
        self,
        email: str,
        password: str,
        device_token: str | None,
        *,
        require_admin: bool = False,
    ) -> tuple[AuthUser, str, str, str]:
        email = self._normalize_email(email)
        self._validate_password(password)
        device_token = device_token or generate_device_token()
        device_hash = token_hash(device_token)
        row = self.repo.user_by_email(email)

        # Use the same public error for unknown email and wrong password.
        if row is None or not verify_password(password, row["password_hash"]):
            raise AuthenticationError("invalid email or password")
        if row["status"] != "active":
            raise ForbiddenError("account is not active", code="account_disabled")
        if require_admin and row["role"] != "admin":
            raise ForbiddenError("admin access is required", code="admin_forbidden")

        if row["role"] == "user" and row["device_hash"] is not None:
            if not hmac.compare_digest(str(row["device_hash"]), device_hash):
                raise ForbiddenError(
                    "this account is already linked to another device",
                    code="device_mismatch",
                    details={"relink_required": True},
                )
        elif row["device_hash"] is None:
            self.repo.bind_device(int(row["id"]), device_hash)

        now = utc_now()
        self.repo.update_last_login(int(row["id"]), now)
        user = self.user_from_row(self.repo.user_by_id(int(row["id"])))
        session_token, csrf_token = self._create_session(user.id, scope="admin" if require_admin else "user")
        self.repo.add_audit(user.id, "admin_login" if require_admin else "login", "user", str(user.id), {})
        return user, session_token, csrf_token, device_token

    def logout(self, raw_token: str | None) -> None:
        if raw_token:
            self.repo.delete_session(raw_token)

    def get_session(self, raw_token: str | None, *, expected_scope: str | None = None) -> SessionContext:
        """Resolve a session token, optionally requiring a specific session scope.

        User and admin sessions live in separate cookies. ``expected_scope``
        guarantees that a token presented in the user area is a ``user`` session
        and a token presented in the admin area is an ``admin`` session, so one
        area can never be authenticated by the other's session.
        """
        area = expected_scope or "any"
        if not raw_token:
            raise AuthenticationError(
                "login required",
                details={"reason": f"no_{area}_session" if expected_scope else "no_session"},
            )
        row = self.repo.session_row(raw_token)
        if row is None:
            raise AuthenticationError(
                "login required",
                details={"reason": f"invalid_{area}_session" if expected_scope else "invalid_session"},
            )
        scope = str(row["scope"])
        if expected_scope is not None and scope != expected_scope:
            # Never touch (or expire) a session that belongs to the other area.
            raise AuthenticationError(
                "login required",
                details={"reason": "session_scope_mismatch", "session_id": int(row["session_id"])},
            )
        expires_at = dt(row["expires_at"])
        if expires_at is None or expires_at <= utc_now():
            self.repo.delete_session(raw_token)
            raise SessionExpiredError(
                "session has expired",
                details={"reason": f"expired_{area}_session" if expected_scope else "expired_session"},
            )
        return SessionContext(
            session_id=int(row["session_id"]),
            user=self.user_from_row(row),
            scope=scope,
            csrf_token="",
            expires_at=expires_at,
        )

    def verify_csrf(
        self,
        raw_token: str | None,
        csrf_token: str | None,
        *,
        expected_scope: str | None = None,
    ) -> SessionContext:
        session = self.get_session(raw_token, expected_scope=expected_scope)
        if not csrf_token:
            raise ForbiddenError("CSRF token is required", code="csrf_required", details={"session_id": session.session_id})
        row = self.repo.session_row(raw_token)
        if row is None or not hmac.compare_digest(str(row["csrf_hash"]), token_hash(csrf_token)):
            raise ForbiddenError("invalid CSRF token", code="csrf_invalid", details={"session_id": session.session_id})
        return SessionContext(
            session_id=session.session_id,
            user=session.user,
            scope=session.scope,
            csrf_token=csrf_token,
            expires_at=session.expires_at,
        )

    def issue_bootstrap_csrf(self) -> str:
        return generate_csrf_token()

    @staticmethod
    def verify_bootstrap_csrf(cookie_token: str | None, header_token: str | None) -> None:
        if not cookie_token or not header_token or not hmac.compare_digest(cookie_token, header_token):
            raise ForbiddenError("invalid CSRF token", code="csrf_invalid")

    def access_state(self, user_id: int) -> AccessState:
        row = self.repo.user_by_id(user_id)
        if row is None or row["status"] != "active":
            return AccessState(False, "unauthorized", "غير مصرح", False, False, None)
        now = utc_now()
        subscription = self.repo.active_subscription(user_id, now)
        trial_expires = dt(row["trial_expires_at"])
        trial_active = bool(trial_expires and trial_expires > now)
        if subscription:
            expiry = dt(subscription["expires_at"])
            return AccessState(True, "active_subscription", "اشتراك فعال", False, True, expiry)
        if trial_active:
            return AccessState(True, "trial_active", "الفترة التجريبية فعالة", True, False, trial_expires)
        return AccessState(False, "subscription_expired", "انتهى الاشتراك أو التجربة", False, False, None)

    def require_analysis_access(self, raw_token: str | None, *, expected_scope: str | None = None) -> SessionContext:
        session = self.get_session(raw_token, expected_scope=expected_scope)
        if session.user.role == "admin":
            return session
        access = self.access_state(session.user.id)
        if not access.allowed:
            raise ForbiddenError(
                "active trial or subscription is required",
                code="subscription_expired",
                details={"whatsapp_cta": "اشترك الآن"},
            )
        return session

    def resolve_admin_session(self, raw_token: str | None, csrf_token: str | None) -> SessionContext:
        """Validate CSRF and reconcile a stale admin-session user id.

        The admin email is the stable identity used by the local bootstrap tool.
        If an administrator account was recreated while an older browser session
        was still alive, the session can momentarily reference the old user id.
        Re-resolve the email and rebind the session to the current admin row before
        any privileged mutation is executed.
        """
        session = self.verify_csrf(raw_token, csrf_token, expected_scope="admin")

        current = self.repo.user_by_email(session.user.email)
        if current is None:
            if raw_token:
                self.logout(raw_token)
            raise AuthenticationError(
                "admin account no longer exists",
                code="admin_session_stale",
                details={"reason": "stale_admin_session", "session_id": session.session_id},
            )

        current_user = self.user_from_row(current)
        if current_user.status != "active" or current_user.role != "admin":
            if raw_token:
                self.logout(raw_token)
            raise ForbiddenError(
                "admin access is required",
                code="admin_forbidden",
                status_code=403,
                details={"reason": "invalid_admin_session", "session_id": session.session_id},
            )

        if current_user.id != session.user.id:
            self.repo.rebind_session_user(session.session_id, current_user.id)

        return SessionContext(
            session_id=session.session_id,
            user=current_user,
            scope=session.scope,
            csrf_token=session.csrf_token,
            expires_at=session.expires_at,
        )

    def redeem_code(self, user_id: int, raw_code: str) -> AccessState:
        try:
            code = normalize_subscription_code(raw_code)
        except ValueError as exc:
            raise ValidationError(str(exc), code="invalid_code") from exc

        now = utc_now()
        code_row = self.repo.code_by_hash(token_hash(code))
        if code_row is None:
            raise ValidationError("subscription code is invalid", code="invalid_code")
        if code_row["status"] == "expired":
            raise ValidationError("subscription code has expired", code="code_expired")
        if code_row["status"] != "available":
            raise ValidationError("subscription code has already been used", code="code_used")
        code_expires_at = dt(code_row["expires_at"])
        if code_expires_at and code_expires_at <= now:
            self.database.execute(
                "UPDATE subscription_codes SET status = 'expired' WHERE id = ? AND status = 'available'",
                (code_row["id"],),
            )
            self.database.commit()
            raise ValidationError("subscription code has expired", code="code_expired")

        if self.repo.user_by_id(user_id) is None:
            raise AuthenticationError(
                "user account no longer exists",
                code="user_session_stale",
                details={"reason": "stale_user_session"},
            )
        connection = self.database.connection
        with self._write_lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                fresh = connection.execute("SELECT * FROM subscription_codes WHERE id = ?", (code_row["id"],)).fetchone()
                if fresh is None or fresh["status"] != "available":
                    connection.rollback()
                    raise ValidationError("subscription code has already been used", code="code_used")
                active = self.repo.active_subscription(user_id, now)
                start_at = dt(active["expires_at"]) if active else now
                expires_at = add_months(start_at or now, int(fresh["duration_months"]))
                connection.execute(
                    """
                    INSERT INTO subscriptions(user_id, starts_at, expires_at, status, source_code_id, created_at)
                    VALUES (?, ?, ?, 'active', ?, ?)
                    """,
                    (user_id, start_at.isoformat(), expires_at.isoformat(), fresh["id"], now.isoformat()),
                )
                connection.execute(
                    "UPDATE subscription_codes SET status = 'redeemed', assigned_user_id = ?, redeemed_at = ? WHERE id = ?",
                    (user_id, now.isoformat(), fresh["id"]),
                )
                connection.execute(
                    "INSERT INTO audit_logs(actor_user_id, action, target_type, target_id, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, "redeem_code", "subscription_code", str(fresh["id"]), json_details({"duration_months": int(fresh["duration_months"])}), now.isoformat()),
                )
                connection.commit()
            except Exception:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
                raise
        return self.access_state(user_id)

    def create_codes(self, admin_user_id: int, duration_months: int, quantity: int) -> list[GeneratedCode]:
        if duration_months not in {1, 3}:
            raise ValidationError("code duration must be 1 or 3 months", code="invalid_duration")
        if quantity < 1 or quantity > 50:
            raise ValidationError("quantity must be between 1 and 50", code="invalid_quantity")

        # Validate the current actor before touching subscription-code state.
        # This converts stale/deleted admin identities into a clean auth error
        # instead of a misleading audit-log foreign-key failure.
        actor = self.repo.user_by_id(admin_user_id)
        if actor is None or actor["role"] != "admin" or actor["status"] != "active":
            raise AuthenticationError("admin account is no longer valid", code="admin_session_stale", status_code=401)

        generated: list[GeneratedCode] = []
        # Code creation + audit are one atomic transaction. The old implementation
        # inserted the code first and then committed through add_audit(), so a
        # foreign-key failure in the audit insert could leave the connection in a
        # partial transaction.
        with self.database.transaction() as connection:
            for _ in range(quantity):
                code = generate_subscription_code()
                code_hint = code[-4:]
                cursor = connection.execute(
                    """
                    INSERT INTO subscription_codes(
                        code_hash, code_hint, duration_months, status, expires_at, created_at
                    ) VALUES (?, ?, ?, 'available', NULL, ?)
                    """,
                    (token_hash(code), code_hint, duration_months, utc_now().isoformat()),
                )
                code_id = int(cursor.lastrowid)
                connection.execute(
                    """
                    INSERT INTO audit_logs(
                        actor_user_id, action, target_type, target_id, details_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        admin_user_id,
                        "generate_code",
                        "subscription_code",
                        str(code_id),
                        json_details({"duration_months": duration_months, "code_hint": code_hint}),
                        utc_now().isoformat(),
                    ),
                )
                generated.append(GeneratedCode(code=code, duration_months=duration_months, expires_at=None))
        return generated

    def revoke_user_subscription(self, admin_user_id: int, user_id: int, reason: str | None) -> bool:
        active = self.repo.active_subscription(user_id)
        if active is None:
            raise ValidationError("user has no active subscription", code="subscription_not_active")
        changed = self.repo.revoke_subscription(int(active["id"]), reason)
        if changed:
            self.repo.add_audit(
                admin_user_id,
                "revoke_subscription",
                "subscription",
                str(active["id"]),
                {"user_id": user_id, "reason": reason},
            )
        return changed

    def extend_user_subscription(self, admin_user_id: int, user_id: int, duration_months: int) -> AccessState:
        if duration_months not in {1, 3}:
            raise ValidationError("extension duration must be 1 or 3 months", code="invalid_duration")
        if self.repo.user_by_id(user_id) is None:
            raise ValidationError("user not found", code="user_not_found")
        now = utc_now()
        active = self.repo.active_subscription(user_id, now)
        if active:
            start_at = dt(active["expires_at"]) or now
            expires_at = add_months(start_at, duration_months)
            self.repo.extend_subscription(int(active["id"]), start_at, expires_at)
            subscription_id = int(active["id"])
        else:
            starts_at = now
            expires_at = add_months(now, duration_months)
            subscription_id = self.repo.create_subscription(user_id, starts_at, expires_at, None)
        self.repo.add_audit(
            admin_user_id,
            "extend_subscription",
            "subscription",
            str(subscription_id),
            {"user_id": user_id, "duration_months": duration_months},
        )
        return self.access_state(user_id)

    def relink_device(self, admin_user_id: int, user_id: int) -> None:
        if self.repo.user_by_id(user_id) is None:
            raise ValidationError("user not found", code="user_not_found")
        self.repo.clear_device(user_id)
        self.repo.delete_user_sessions(user_id)
        self.repo.add_audit(admin_user_id, "relink_device", "user", str(user_id), {"mode": "reset_for_next_login"})

    def user_payload(self, user: AuthUser) -> dict[str, Any]:
        access = self.access_state(user.id)
        return {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "status": user.status,
            "trial_started_at": user.trial_started_at.isoformat(),
            "trial_expires_at": user.trial_expires_at.isoformat(),
            "created_at": user.created_at.isoformat(),
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
            "device_bound": user.device_bound,
            "access": {
                "allowed": access.allowed,
                "code": access.code,
                "label_ar": access.label_ar,
                "trial_active": access.trial_active,
                "subscription_active": access.subscription_active,
                "expires_at": access.expires_at.isoformat() if access.expires_at else None,
                "cta": "اشترك الآن" if not access.allowed else None,
            },
        }

    @staticmethod
    def user_from_row(row: sqlite3.Row | None) -> AuthUser:
        if row is None:
            raise AuthenticationError("user not found")
        return AuthUser(
            id=int(row["id"]),
            email=str(row["email"]),
            role=str(row["role"]),
            status=str(row["status"]),
            trial_started_at=dt(row["trial_started_at"]) or utc_now(),
            trial_expires_at=dt(row["trial_expires_at"]) or utc_now(),
            created_at=dt(row["created_at"]) or utc_now(),
            last_login_at=dt(row["last_login_at"]),
            device_bound=row["device_hash"] is not None,
        )

    def _create_session(self, user_id: int, *, scope: str) -> tuple[str, str]:
        raw_token = generate_device_token()
        csrf_token = generate_csrf_token()
        expires_at = utc_now() + self.session_ttl
        self.repo.create_session(
            user_id,
            scope=scope,
            raw_token=raw_token,
            csrf_token=csrf_token,
            expires_at=expires_at,
        )
        return raw_token, csrf_token

    @staticmethod
    def _normalize_email(value: str) -> str:
        try:
            return normalize_email(value)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    @staticmethod
    def _validate_password(value: str) -> str:
        try:
            return validate_password(value)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc


def add_months(value: datetime, months: int) -> datetime:
    """Add calendar months without drifting at month ends."""
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def json_details(details: dict[str, Any]) -> str:
    import json

    return json.dumps(details, ensure_ascii=False, sort_keys=True)

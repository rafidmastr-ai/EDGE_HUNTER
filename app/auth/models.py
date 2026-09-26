"""Domain models shared by authentication and administration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AuthUser:
    id: int
    email: str
    role: str
    status: str
    trial_started_at: datetime
    trial_expires_at: datetime
    created_at: datetime
    last_login_at: datetime | None
    device_bound: bool


@dataclass(frozen=True)
class SessionContext:
    session_id: int
    user: AuthUser
    scope: str
    csrf_token: str
    expires_at: datetime


@dataclass(frozen=True)
class AccessState:
    allowed: bool
    code: str
    label_ar: str
    trial_active: bool
    subscription_active: bool
    expires_at: datetime | None


@dataclass(frozen=True)
class GeneratedCode:
    code: str
    duration_months: int
    expires_at: datetime | None

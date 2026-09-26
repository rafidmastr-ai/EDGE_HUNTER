"""Administration read/write façade over the authentication service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.auth.repository import AuthRepository, dt
from app.auth.service import AuthService


class AdminService:
    """Builds safe admin dashboard payloads and delegates mutations."""

    def __init__(self, auth: AuthService) -> None:
        self.auth = auth
        self.repo = AuthRepository(auth.database)

    def dashboard(self) -> dict[str, Any]:
        return {
            "counts": self.repo.dashboard_counts(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def users(self) -> list[dict[str, Any]]:
        result = []
        for row in self.repo.list_users():
            user = self.auth.user_from_row(row)
            access = self.auth.access_state(user.id)
            result.append(
                {
                    "id": user.id,
                    "email": user.email,
                    "role": user.role,
                    "status": user.status,
                    "trial_expires_at": user.trial_expires_at.isoformat(),
                    "device_bound": user.device_bound,
                    "access_code": access.code,
                    "access_label_ar": access.label_ar,
                    "access_expires_at": access.expires_at.isoformat() if access.expires_at else None,
                    "subscriptions": [
                        {
                            "id": int(sub["id"]),
                            "starts_at": sub["starts_at"],
                            "expires_at": sub["expires_at"],
                            "status": sub["status"],
                        }
                        for sub in self.repo.subscriptions_for_user(user.id)
                    ],
                }
            )
        return result

    def codes(self) -> list[dict[str, Any]]:
        output = []
        for row in self.repo.list_codes():
            output.append(
                {
                    "id": int(row["id"]),
                    "code_hint": row["code_hint"],
                    "duration_months": int(row["duration_months"]),
                    "status": row["status"],
                    "expires_at": row["expires_at"],
                    "assigned_email": row["assigned_email"],
                    "redeemed_at": row["redeemed_at"],
                    "created_at": row["created_at"],
                }
            )
        return output

    def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "id": int(row["id"]),
                "actor_email": row["actor_email"],
                "action": row["action"],
                "target_type": row["target_type"],
                "target_id": row["target_id"],
                "details_json": row["details_json"],
                "created_at": row["created_at"],
            }
            for row in self.repo.list_audit(limit)
        ]

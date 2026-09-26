"""Pydantic contracts for administration endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.auth.security import normalize_email


class AdminLoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email", mode="before")
    @classmethod
    def normalize(cls, value: str) -> str:
        return normalize_email(str(value))


class CodeGenerationRequest(BaseModel):
    duration_months: int = Field(default=1)
    quantity: int = Field(default=1, ge=1, le=50)

    @field_validator("duration_months")
    @classmethod
    def duration_allowed(cls, value: int) -> int:
        if value not in {1, 3}:
            raise ValueError("duration_months must be 1 or 3")
        return value


class SubscriptionActionRequest(BaseModel):
    duration_months: int = Field(default=1)
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("duration_months")
    @classmethod
    def duration_allowed(cls, value: int) -> int:
        if value not in {1, 3}:
            raise ValueError("duration_months must be 1 or 3")
        return value

"""Pydantic contracts for the public authentication API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.auth.security import normalize_email


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    password_confirm: str = Field(min_length=8, max_length=128)

    @field_validator("email", mode="before")
    @classmethod
    def normalize(cls, value: str) -> str:
        return normalize_email(str(value))

    @field_validator("password_confirm")
    @classmethod
    def passwords_match(cls, value: str, info) -> str:
        password = info.data.get("password")
        if password is not None and value != password:
            raise ValueError("passwords do not match")
        return value


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email", mode="before")
    @classmethod
    def normalize(cls, value: str) -> str:
        return normalize_email(str(value))


class RedeemCodeRequest(BaseModel):
    code: str = Field(min_length=16, max_length=16)


class AuthUserResponse(BaseModel):
    id: int
    email: str
    role: str
    status: str
    trial_started_at: str
    trial_expires_at: str
    created_at: str
    last_login_at: str | None
    device_bound: bool
    access: dict

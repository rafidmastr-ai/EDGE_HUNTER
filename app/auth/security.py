"""Security helpers for passwords, sessions, CSRF and device binding.

Only hashes of passwords, session tokens, CSRF tokens and device tokens are
persisted. Raw credentials/tokens exist only for the lifetime required to set a
cookie or authenticate one request.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import unicodedata

PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 600_000
PASSWORD_SALT_BYTES = 16
PASSWORD_DK_BYTES = 32
TOKEN_BYTES = 32
EMAIL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*/=?^_`{|}~+/-]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,63}$")
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def normalize_email(value: str) -> str:
    """Normalize an email address for stable account lookup."""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().lower()
    if len(normalized) > 320 or not EMAIL_RE.fullmatch(normalized):
        raise ValueError("invalid email address")
    return normalized


def validate_password(value: str) -> str:
    """Validate a user/admin password without storing it."""
    password = str(value)
    if len(password) < 8:
        raise ValueError("password must contain at least 8 characters")
    if len(password) > 128:
        raise ValueError("password must not exceed 128 characters")
    return password


def hash_password(password: str, iterations: int = PASSWORD_ITERATIONS) -> str:
    """Return a self-describing PBKDF2-HMAC-SHA256 password hash."""
    password = validate_password(password)
    salt = os.urandom(PASSWORD_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, PASSWORD_DK_BYTES)
    return "$".join(
        (
            PASSWORD_SCHEME,
            str(iterations),
            _b64(salt),
            _b64(digest),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    """Verify a password against a stored self-describing hash."""
    try:
        scheme, raw_iterations, raw_salt, raw_expected = encoded.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        iterations = int(raw_iterations)
        salt = _unb64(raw_salt)
        expected = _unb64(raw_expected)
        if iterations < 100_000 or iterations > 2_000_000:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, iterations, len(expected))
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError, base64.binascii.Error):
        return False


def random_token() -> str:
    """Generate a high-entropy URL-safe token."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_hash(token: str) -> str:
    """Hash a bearer token before persisting it."""
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    """Generate a per-session CSRF token."""
    return random_token()


def generate_device_token() -> str:
    """Generate a persistent browser-device token."""
    return random_token()


def generate_subscription_code(length: int = 16) -> str:
    """Generate an uppercase, human-enterable alphanumeric subscription code."""
    if length != 16:
        raise ValueError("subscription codes must be exactly 16 characters")
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(length))


def normalize_subscription_code(value: str) -> str:
    """Normalize a code for hashing while preserving no secret whitespace."""
    code = "".join(str(value).strip().split()).upper()
    if len(code) != 16 or any(char not in CODE_ALPHABET for char in code):
        raise ValueError("subscription code must be 16 alphanumeric characters")
    return code


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

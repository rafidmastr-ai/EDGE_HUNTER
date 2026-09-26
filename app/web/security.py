"""Public-edge security middleware for EDGE HUNTER Phase 11."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Awaitable, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.auth.rate_limit import RateLimiter


class RequestTooLargeError(Exception):
    """Internal signal raised when an HTTP request exceeds the configured body limit."""


class JsonFormatter(logging.Formatter):
    """Small JSON formatter that never serializes request bodies or headers."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status", "duration_ms", "client_ip"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_request_logging(log_dir: str) -> logging.Logger:
    """Configure a dedicated structured request logger once per process."""
    logger = logging.getLogger("edge_hunter.request")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = RotatingFileHandler(
        path / "requests.jsonl",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def client_ip(scope: Scope, *, trust_proxy_headers: bool) -> str:
    """Return a rate-limit key IP without trusting spoofable proxy headers by default."""
    if trust_proxy_headers:
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])}
        forwarded = headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",", 1)[0].strip() or "unknown"
        real_ip = headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip() or "unknown"
        cf_ip = headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip() or "unknown"
    peer = scope.get("client")
    return peer[0] if peer else "unknown"


async def _send_json(send: Send, status_code: int, payload: dict, headers: dict[str, str] | None = None) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    response_headers = [(b"content-type", b"application/json; charset=utf-8"), (b"content-length", str(len(raw)).encode())]
    for key, value in (headers or {}).items():
        response_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))
    await send({"type": "http.response.start", "status": status_code, "headers": response_headers})
    await send({"type": "http.response.body", "body": raw})


class RequestIDLoggingMiddleware:
    """Attach request IDs and write structured access/error logs."""

    def __init__(self, app: ASGIApp, *, log_dir: str, trust_proxy_headers: bool) -> None:
        self.app = app
        self.logger = configure_request_logging(log_dir)
        self.trust_proxy_headers = trust_proxy_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500
        response_started = False
        peer_ip = client_ip(scope, trust_proxy_headers=self.trust_proxy_headers)
        method = scope.get("method", "?")
        path = scope.get("path", "?")

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                response_started = True
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except RequestTooLargeError:
            if not response_started:
                await _send_json(
                    send,
                    413,
                    {"detail": {"code": "request_too_large", "message": "request body is too large", "request_id": request_id}},
                )
                status_code = 413
            return
        except Exception:
            self.logger.error(
                "Unhandled application exception",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status": 500,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "client_ip": peer_ip,
                },
            )
            if not response_started:
                await _send_json(
                    send,
                    500,
                    {"detail": {"code": "internal_error", "message": "internal server error", "request_id": request_id}},
                )
            return

        self.logger.info(
            "HTTP request",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "status": status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "client_ip": peer_ip,
            },
        )


class RequestSizeLimitMiddleware:
    """Reject oversized mutating requests before JSON parsing."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max(1_024, int(max_bytes))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        content_length = headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await _send_json(send, 413, {"detail": {"code": "request_too_large", "message": "request body is too large"}})
                    return
            except ValueError:
                await _send_json(send, 400, {"detail": {"code": "invalid_content_length", "message": "invalid request"}})
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                received += len(body)
                if received > self.max_bytes:
                    raise RequestTooLargeError()
            return message

        await self.app(scope, limited_receive, send)


class PublicRateLimitMiddleware:
    """Apply a modest process-local API rate limit before route execution."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: RateLimiter,
        limit: int,
        window_seconds: int,
        trust_proxy_headers: bool,
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self.trust_proxy_headers = trust_proxy_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or not path.startswith("/api/") or path in {"/api/health", "/api/ready"}:
            await self.app(scope, receive, send)
            return

        ip = client_ip(scope, trust_proxy_headers=self.trust_proxy_headers)
        key = f"public:{ip}"
        try:
            self.limiter.check(key, limit=self.limit, window_seconds=self.window_seconds)
        except Exception as exc:
            if getattr(exc, "code", None) == "rate_limited":
                retry_after = str(self.window_seconds)
                await _send_json(
                    send,
                    429,
                    {"detail": {"code": "rate_limited", "message": "too many requests; please try again later"}},
                    headers={"Retry-After": retry_after, "Cache-Control": "no-store"},
                )
                return
            raise
        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Add browser security headers without exposing implementation details."""

    def __init__(self, app: ASGIApp, *, production: bool) -> None:
        self.app = app
        self.production = production

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {key.lower() for key, _ in headers}
                security_headers = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    (
                        b"content-security-policy",
                        b"default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'; form-action 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'",
                    ),
                ]
                if self.production:
                    security_headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                if scope.get("path", "").startswith("/api/"):
                    security_headers.append((b"cache-control", b"no-store"))
                for key, value in security_headers:
                    if key not in existing:
                        headers.append((key, value))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)

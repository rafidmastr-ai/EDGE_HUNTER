"""FastAPI application for EDGE HUNTER Phase 11.

Phase 11 keeps the Phase 10 authentication/subscription stack and adds
public-edge security, observability and production controls.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from app.admin.schemas import AdminLoginRequest, CodeGenerationRequest, SubscriptionActionRequest
from app.admin.service import AdminService
from app.auth.rate_limit import RateLimiter
from app.auth.schemas import LoginRequest, RedeemCodeRequest, RegisterRequest
from app.auth.service import AuthError, AuthService
from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.web.analysis_service import DataUnavailableError, LiveDataUnavailableError, LocalOHLCAnalysisService
from app.web.admin_code_diagnostics import save_admin_code_generation_diagnostic
from app.web.schemas import AnalysisResponse, AnalyzeRequest, DEFAULT_SYMBOLS, RISK_PRESETS
from app.web.symbol_catalog import SymbolCatalogService, SymbolCatalogUnavailableError
from app.web.security import (
    PublicRateLimitMiddleware,
    RequestIDLoggingMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from config.config_hunter import Settings, load_settings


APP_VERSION = "phase13-v1"
SESSION_COOKIE = "eh_session"
CSRF_COOKIE = "eh_csrf"
BOOTSTRAP_CSRF_COOKIE = "eh_bootstrap_csrf"
DEVICE_COOKIE = "eh_device"


def create_web_app(
    data_root: Path | None = None,
    *,
    database: Database | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create a web application with injectable data/database/settings for tests."""
    settings = settings or load_settings()
    database = database or Database(settings.database_path)
    MigrationRunner(database).apply_all()
    auth = AuthService(
        database,
        session_ttl_hours=settings.session_ttl_hours,
        password_iterations=settings.password_iterations,
    )
    admin = AdminService(auth)
    limiter = RateLimiter()
    from app.providers.factory import build_live_provider
    live_provider = build_live_provider(settings)

    app = FastAPI(
        title="EDGE HUNTER",
        version=APP_VERSION,
        docs_url="/api/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.docs_enabled else None,
    )
    app.state.database = database
    app.state.auth = auth
    app.state.admin = admin
    app.state.rate_limiter = limiter
    app.state.settings = settings
    app.state.live_provider = live_provider
    symbol_catalog = SymbolCatalogService(
        live_provider,
        cache_ttl_seconds=settings.symbol_catalog_cache_ttl_seconds,
        max_results=settings.symbol_search_limit,
    )
    app.state.symbol_catalog = symbol_catalog

    # Public-edge controls are installed before route execution.
    # add_middleware() inserts at the front, so the final stack is:
    # request logging -> CORS -> security headers -> size limit -> rate limit -> trusted host -> FastAPI.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(
        PublicRateLimitMiddleware,
        limiter=limiter,
        limit=settings.public_rate_limit,
        window_seconds=settings.public_rate_limit_window_seconds,
        trust_proxy_headers=settings.trust_proxy_headers,
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(SecurityHeadersMiddleware, production=settings.environment == "production")
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Accept", "Content-Type", "X-CSRF-Token"],
            max_age=600,
        )
    app.add_middleware(
        RequestIDLoggingMiddleware,
        log_dir=str(settings.log_dir),
        trust_proxy_headers=settings.trust_proxy_headers,
    )

    @app.exception_handler(AuthError)
    async def auth_error_handler(request: Request, exc: AuthError) -> Response:
        # Keep all auth errors in one structured, non-secret API contract.
        return Response(
            content=json.dumps({"detail": {"code": exc.code, "message": str(exc), **exc.details, "request_id": getattr(request.state, "request_id", None)}}, ensure_ascii=False),
            status_code=exc.status_code,
            media_type="application/json",
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for item in exc.errors():
            errors.append({"loc": list(item.get("loc", ())), "msg": str(item.get("msg", "invalid input")), "type": str(item.get("type", "validation_error"))})
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "validation_error",
                    "message": "invalid request payload",
                    "errors": errors,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            safe_detail = dict(detail)
        else:
            safe_detail = {"code": "http_error", "message": "request failed"}
        safe_detail.setdefault("request_id", getattr(request.state, "request_id", None))
        return JSONResponse(status_code=exc.status_code, content={"detail": safe_detail})

    service = LocalOHLCAnalysisService(
        data_root=data_root,
        live_provider=live_provider,
        data_mode=settings.data_mode,
        live_history_bars=settings.live_provider_max_bars,
        live_fallback_to_local=settings.live_fallback_to_local and settings.environment != "production",
    )
    app.state.analysis_service = service
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/admin", include_in_schema=False)
    async def admin_index() -> FileResponse:
        return FileResponse(static_dir / "admin.html")

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "service": "edge-hunter-web", "version": APP_VERSION}

    @app.get("/api/ready")
    async def ready() -> Response:
        try:
            database.execute("SELECT 1").fetchone()
            live_health_payload = service.live_health()
            if settings.data_mode == "live" and settings.environment == "production" and not live_health_payload.get("configured", False):
                return JSONResponse(
                    status_code=503,
                    content={"status": "not_ready", "service": "edge-hunter-web", "version": APP_VERSION, "reason": "live_provider_unconfigured"},
                    headers={"Cache-Control": "no-store"},
                )
            return JSONResponse({"status": "ready", "service": "edge-hunter-web", "version": APP_VERSION})
        except Exception:
            logging.getLogger("edge_hunter.request").error("Readiness check failed", exc_info=True)
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "service": "edge-hunter-web", "version": APP_VERSION},
                headers={"Cache-Control": "no-store"},
            )

    @app.get("/api/live-health")
    async def live_health() -> dict:
        """Return safe live-provider health without secrets or upstream payloads."""
        health_payload = service.live_health()
        health_payload["data_mode"] = settings.data_mode
        health_payload["fallback_to_local"] = service.live_fallback_to_local
        return health_payload

    @app.get("/api/meta")
    async def meta() -> dict:
        return {
            "symbols": list(DEFAULT_SYMBOLS),
            "symbol_catalog_endpoint": "/api/symbols/search",
            "symbol_categories": ["all", "forex", "crypto", "metals"],
            "risk_presets": list(RISK_PRESETS),
            "lot_modes": ["auto", "manual"],
            "timeframe_control": "system_selected",
            "supports_capital": True,
            "auth_required_for_analysis": True,
            "subscription_code_length": 16,
            "subscription_durations_months": [1, 3],
            "whatsapp_url": settings.whatsapp_url,
            "data_mode": settings.data_mode,
            "live_provider": service.live_health(),
        }

    @app.get("/api/symbols/search")
    async def symbol_search(q: str = "", category: str = "all", limit: int = 20) -> dict:
        """Search the provider-supported Forex/Crypto/Metal symbols for the Web UI."""
        try:
            return symbol_catalog.search(query=q, category=category, limit=limit)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_symbol_search", "message": str(exc)},
            ) from exc
        except SymbolCatalogUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": exc.code, "message": "market symbol catalog is currently unavailable"},
            ) from exc

    @app.get("/api/auth/csrf")
    async def csrf(response: Response) -> dict:
        token = auth.issue_bootstrap_csrf()
        _set_cookie(
            response,
            BOOTSTRAP_CSRF_COOKIE,
            token,
            httponly=False,
            secure=settings.cookie_secure,
        )
        return {"csrf_token": token}

    @app.post("/api/auth/register")
    async def register(payload: RegisterRequest, request: Request, response: Response) -> dict:
        _check_bootstrap_csrf(request, auth)
        _limit(limiter, request, "register", settings)
        user, session_token, csrf_token, device_token = auth.register(
            payload.email,
            payload.password,
            request.cookies.get(DEVICE_COOKIE),
        )
        _set_auth_cookies(response, session_token, csrf_token, device_token, settings)
        return {"user": auth.user_payload(user), "csrf_token": csrf_token}

    @app.post("/api/auth/login")
    async def login(payload: LoginRequest, request: Request, response: Response) -> dict:
        _check_bootstrap_csrf(request, auth)
        _limit(limiter, request, "login", settings)
        user, session_token, csrf_token, device_token = auth.login(
            payload.email,
            payload.password,
            request.cookies.get(DEVICE_COOKIE),
        )
        _set_auth_cookies(response, session_token, csrf_token, device_token, settings)
        return {"user": auth.user_payload(user), "csrf_token": csrf_token}

    @app.post("/api/admin/login")
    async def admin_login(payload: AdminLoginRequest, request: Request, response: Response) -> dict:
        _check_bootstrap_csrf(request, auth)
        _limit(limiter, request, "admin_login", settings)
        user, session_token, csrf_token, device_token = auth.login(
            payload.email,
            payload.password,
            request.cookies.get(DEVICE_COOKIE),
            require_admin=True,
        )
        _set_auth_cookies(response, session_token, csrf_token, device_token, settings)
        return {"user": auth.user_payload(user), "csrf_token": csrf_token}

    @app.get("/api/auth/status")
    async def auth_status(request: Request) -> dict:
        """Return a non-error authentication state for public-page bootstrapping.

        A logged-out browser should not need to request the protected /me endpoint
        merely to discover that it is logged out. Returning 200 here avoids expected
        401 resource errors in browser consoles while keeping /api/auth/me protected.
        """
        raw_token = request.cookies.get(SESSION_COOKIE)
        if not raw_token:
            return {"authenticated": False, "reason": "anonymous", "user": None}
        try:
            session = auth.get_session(raw_token)
        except AuthError as exc:
            return {"authenticated": False, "reason": exc.code, "user": None}
        return {
            "authenticated": True,
            "reason": "authenticated",
            "user": auth.user_payload(session.user),
            "session_expires_at": session.expires_at.isoformat(),
        }

    @app.get("/api/auth/me")
    async def me(request: Request) -> dict:
        session = auth.get_session(request.cookies.get(SESSION_COOKIE))
        return {"user": auth.user_payload(session.user), "session_expires_at": session.expires_at.isoformat()}

    @app.post("/api/auth/logout")
    async def logout(request: Request, response: Response) -> dict:
        auth.verify_csrf(request.cookies.get(SESSION_COOKIE), request.headers.get("X-CSRF-Token"))
        auth.logout(request.cookies.get(SESSION_COOKIE))
        _clear_auth_cookies(response, settings)
        return {"status": "ok"}

    @app.post("/api/auth/redeem")
    async def redeem(payload: RedeemCodeRequest, request: Request) -> dict:
        session = auth.verify_csrf(request.cookies.get(SESSION_COOKIE), request.headers.get("X-CSRF-Token"))
        if session.user.role != "user":
            raise _http_error(AuthError("user account required", code="user_required", status_code=403))
        _limit(limiter, request, "redeem", settings)
        state = auth.redeem_code(session.user.id, payload.code)
        return {
            "status": state.code,
            "allowed": state.allowed,
            "label_ar": state.label_ar,
            "trial_active": state.trial_active,
            "subscription_active": state.subscription_active,
            "expires_at": state.expires_at.isoformat() if state.expires_at else None,
        }

    @app.post("/api/analyze", response_model=AnalysisResponse)
    async def analyze(payload: AnalyzeRequest, request: Request) -> dict:
        auth.verify_csrf(request.cookies.get(SESSION_COOKIE), request.headers.get("X-CSRF-Token"))
        auth.require_analysis_access(request.cookies.get(SESSION_COOKIE))
        try:
            return service.analyze(payload)
        except LiveDataUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": exc.code, "message": "live market data is currently unavailable"},
            ) from exc
        except DataUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "data_unavailable", "message": str(exc)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "api_error", "message": str(exc)},
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive API boundary
            raise HTTPException(
                status_code=500,
                detail={"code": "api_error", "message": "analysis failed"},
            ) from exc

    @app.get("/api/admin/dashboard")
    async def admin_dashboard(request: Request) -> dict:
        _admin_session(request, auth)
        return admin.dashboard()

    @app.get("/api/admin/users")
    async def admin_users(request: Request) -> dict:
        _admin_session(request, auth)
        return {"users": admin.users()}

    @app.get("/api/admin/codes")
    async def admin_codes(request: Request) -> dict:
        _admin_session(request, auth)
        return {"codes": admin.codes()}

    @app.get("/api/admin/audit")
    async def admin_audit(request: Request, limit: int = 100) -> dict:
        _admin_session(request, auth)
        return {"audit": admin.audit(limit)}

    @app.post("/api/admin/codes")
    async def generate_codes(payload: CodeGenerationRequest, request: Request) -> dict:
        session = _admin_mutation_session(request, auth)
        try:
            generated = auth.create_codes(session.user.id, payload.duration_months, payload.quantity)
        except Exception as exc:
            diagnostic_path = None
            try:
                diagnostic_path = save_admin_code_generation_diagnostic(
                    database=database,
                    settings=settings,
                    request=request,
                    admin_user_id=session.user.id,
                    payload=payload,
                    exc=exc,
                )
            except Exception:
                logging.getLogger("edge_hunter.request").exception(
                    "Admin code generation failed and automatic diagnostic could not be written",
                    extra={"request_id": getattr(request.state, "request_id", None)},
                )
            finally:
                try:
                    database.connection.rollback()
                except sqlite3.Error:
                    pass

            request_id = getattr(request.state, "request_id", None)
            logging.getLogger("edge_hunter.request").error(
                "Admin code generation failed; automatic diagnostic attempted",
                extra={
                    "request_id": request_id,
                    "diagnostic_path": str(diagnostic_path) if diagnostic_path else None,
                    "exception_type": type(exc).__name__,
                },
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "admin_code_generation_failed",
                    "message": "تعذر توليد الكود. تم حفظ تقرير التشخيص تلقائيًا في مجلد logs.",
                    "request_id": request_id,
                    "diagnostic_saved": diagnostic_path is not None,
                },
            ) from exc
        return {
            "codes": [
                {"code": item.code, "duration_months": item.duration_months, "expires_at": item.expires_at}
                for item in generated
            ],
            "warning": "احفظ الأكواد فورًا؛ لا يتم تخزين النص الخام للكود.",
        }

    @app.post("/api/admin/users/{user_id}/revoke")
    async def revoke_user(user_id: int, payload: SubscriptionActionRequest, request: Request) -> dict:
        session = _admin_mutation_session(request, auth)
        changed = auth.revoke_user_subscription(session.user.id, user_id, payload.reason)
        return {"status": "revoked" if changed else "not_changed"}

    @app.post("/api/admin/users/{user_id}/extend")
    async def extend_user(user_id: int, payload: SubscriptionActionRequest, request: Request) -> dict:
        session = _admin_mutation_session(request, auth)
        access = auth.extend_user_subscription(session.user.id, user_id, payload.duration_months)
        return {
            "status": access.code,
            "allowed": access.allowed,
            "expires_at": access.expires_at.isoformat() if access.expires_at else None,
        }

    @app.post("/api/admin/users/{user_id}/relink")
    async def relink_user(user_id: int, request: Request) -> dict:
        session = _admin_mutation_session(request, auth)
        auth.relink_device(session.user.id, user_id)
        return {"status": "relink_ready"}

    @app.post("/api/admin/codes/{code_id}/revoke")
    async def revoke_code(code_id: int, request: Request) -> dict:
        session = _admin_mutation_session(request, auth)
        row = auth.repo.database.execute(
            "UPDATE subscription_codes SET status = 'revoked' WHERE id = ? AND status = 'available'",
            (code_id,),
        )
        auth.database.commit()
        if row.rowcount != 1:
            raise _http_error(AuthError("code is not available", code="code_not_available", status_code=409))
        auth.repo.add_audit(session.user.id, "revoke_code", "subscription_code", str(code_id), {})
        return {"status": "revoked"}

    return app


def _limit(limiter: RateLimiter, request: Request, action: str, settings: Settings) -> None:
    client_ip = request.client.host if request.client else "unknown"
    limiter.check(
        f"{action}:{client_ip}",
        limit=settings.login_rate_limit,
        window_seconds=settings.rate_limit_window_seconds,
    )


def _admin_session(request: Request, auth: AuthService):
    session = auth.get_session(request.cookies.get(SESSION_COOKIE))
    if session.user.role != "admin" or session.scope != "admin":
        raise _http_error(AuthError("admin access is required", code="admin_forbidden", status_code=403))
    return session


def _admin_mutation_session(request: Request, auth: AuthService):
    # Use the reconciliation-aware admin session resolver for every privileged
    # mutation. This is important when the browser keeps a session created
    # before an admin account was recreated/restored with a new user id.
    try:
        return auth.resolve_admin_session(
            request.cookies.get(SESSION_COOKIE),
            request.headers.get("X-CSRF-Token"),
        )
    except AuthError as exc:
        raise _http_error(exc) from exc


def _check_bootstrap_csrf(request: Request, auth: AuthService) -> None:
    try:
        auth.verify_bootstrap_csrf(
            request.cookies.get(BOOTSTRAP_CSRF_COOKIE),
            request.headers.get("X-CSRF-Token"),
        )
    except AuthError as exc:
        raise _http_error(exc) from exc


def _set_auth_cookies(response: Response, session_token: str, csrf_token: str, device_token: str, settings: Settings) -> None:
    _set_cookie(response, SESSION_COOKIE, session_token, httponly=True, secure=settings.cookie_secure)
    _set_cookie(response, CSRF_COOKIE, csrf_token, httponly=False, secure=settings.cookie_secure)
    _set_cookie(response, DEVICE_COOKIE, device_token, httponly=True, secure=settings.cookie_secure, max_age=31536000)


def _set_cookie(response: Response, name: str, value: str, *, httponly: bool, secure: bool, max_age: int | None = None) -> None:
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        httponly=httponly,
        secure=secure,
        samesite="lax",
        path="/",
    )


def _clear_auth_cookies(response: Response, settings: Settings) -> None:
    for name in (SESSION_COOKIE, CSRF_COOKIE, BOOTSTRAP_CSRF_COOKIE):
        response.delete_cookie(name, path="/", secure=settings.cookie_secure, httponly=name == SESSION_COOKIE)


def _http_error(exc: AuthError) -> HTTPException:
    detail: dict[str, Any] = {"code": exc.code, "message": str(exc)}
    detail.update(exc.details)
    return HTTPException(status_code=exc.status_code, detail=detail)


app = create_web_app()

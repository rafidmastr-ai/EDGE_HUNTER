"""FastAPI application for EDGE HUNTER Phase 11.

Phase 11 keeps the Phase 10 authentication/subscription stack and adds
public-edge security, observability and production controls.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.admin.schemas import AdminLoginRequest, CodeGenerationRequest, SubscriptionActionRequest
from app.admin.service import AdminService
from app.auth.rate_limit import RateLimiter
from app.auth.repository import utc_now
from app.auth.schemas import LoginRequest, RedeemCodeRequest, RegisterRequest
from app.auth.service import AuthError, AuthService
from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.strategy_learning import LiveSignalRecorder, StrategyLearningFilter
from app.signals.costs import CostGate
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
from config.config_hunter import PROJECT_ROOT, Settings, load_settings


APP_VERSION = "phase13-v1"
# User and admin areas use fully independent cookies so that both can be signed
# in from the same browser at the same time without replacing each other.
SESSION_COOKIE = "eh_session"
CSRF_COOKIE = "eh_csrf"
BOOTSTRAP_CSRF_COOKIE = "eh_bootstrap_csrf"
ADMIN_SESSION_COOKIE = "eh_admin_session"
ADMIN_CSRF_COOKIE = "eh_admin_csrf"
ADMIN_BOOTSTRAP_CSRF_COOKIE = "eh_admin_bootstrap_csrf"
DEVICE_COOKIE = "eh_device"
USER_SCOPE = "user"
ADMIN_SCOPE = "admin"
# Diagnostic-only fields: logged server-side, never returned to the client.
_PRIVATE_ERROR_DETAILS = frozenset({"session_id"})

auth_logger = logging.getLogger("edge_hunter.auth")


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
        _log_auth_failure(request, "request", exc)
        return Response(
            content=json.dumps({"detail": {"code": exc.code, "message": str(exc), **_public_details(exc), "request_id": getattr(request.state, "request_id", None)}}, ensure_ascii=False),
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

    # Learned per-strategy filters (read-only; trained by scripts/train_strategies.py)
    # and the post-response recorder that feeds live setups back into training.
    strategy_filter = StrategyLearningFilter(settings.strategy_ml_model_path, enabled=settings.strategy_ml_enabled)
    live_signal_recorder = LiveSignalRecorder(database, enabled=settings.strategy_ml_record_live)
    # Experimental EDGE ML paper signals: own card and paper-trade log, never the main signal.
    edge_ml = None
    if settings.edge_ml_enabled:
        from app.ml_edge.live import SYMBOLS as EDGE_ML_SYMBOLS, EdgeMLScheduler, EdgeMLService
        from app.ml_edge.live_store import M1Store

        edge_ml = EdgeMLService(
            settings.edge_ml_models_dir,
            M1Store(settings.edge_ml_store_dir, EDGE_ML_SYMBOLS, raw_dir=data_root or (PROJECT_ROOT / "data" / "raw")),
            database,
            live_provider,
            # a decision stays actionable until the next scheduled refresh (+5 min margin)
            fresh_seconds=max(20 * 60, settings.edge_ml_refresh_minutes * 60 + 5 * 60),
            daily_request_budget=settings.edge_ml_daily_request_budget,
        )
        # Only a configured live provider feeds the store; the test environment and local mode never start the thread.
        if (settings.data_mode == "live" and settings.environment != "test"
                and live_provider is not None and getattr(live_provider, "configured", False)):
            _run_with_app(app, EdgeMLScheduler(edge_ml, interval_minutes=settings.edge_ml_refresh_minutes))
    service = LocalOHLCAnalysisService(
        data_root=data_root,
        live_provider=live_provider,
        data_mode=settings.data_mode,
        live_history_bars=settings.live_provider_max_bars,
        # Learned filters apply to live analysis only; "local" is the explicit
        # offline/test mode and stays independent of locally trained model files.
        signal_filter=strategy_filter if settings.data_mode == "live" else None,
        # Cost-aware gate is a deterministic rule (not a learned model): both modes.
        cost_gate=CostGate(max_cost_r=settings.cost_gate_max_cost_r) if settings.cost_gate_enabled else None,
        edge_ml=edge_ml,
    )
    app.state.analysis_service = service
    app.state.edge_ml = edge_ml
    app.state.strategy_filter = strategy_filter
    app.state.live_signal_recorder = live_signal_recorder
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        # versioned asset URLs + no-cache: after copying new files the browser never mixes an
        # old index.html with a new app.js (or the reverse)
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        for asset in ("app.js", "styles.css"):
            stat = (static_dir / asset).stat()
            html = html.replace(f'"/static/{asset}"', f'"/static/{asset}?v={int(stat.st_mtime)}-{stat.st_size}"')
        return Response(content=html, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})

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
        # Live analysis never falls back to local CSV data.
        health_payload["fallback_to_local"] = False
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
            "strategy_learning": service.strategy_learning_status(),
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

    @app.get("/api/admin/csrf")
    async def admin_csrf(response: Response) -> dict:
        # Separate bootstrap cookie: the user page refreshing its own bootstrap
        # token must not invalidate the token held by an open admin login form.
        token = auth.issue_bootstrap_csrf()
        _set_cookie(
            response,
            ADMIN_BOOTSTRAP_CSRF_COOKIE,
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
        _check_admin_bootstrap_csrf(request, auth)
        _limit(limiter, request, "admin_login", settings)
        user, session_token, csrf_token, device_token = auth.login(
            payload.email,
            payload.password,
            request.cookies.get(DEVICE_COOKIE),
            require_admin=True,
        )
        # Admin cookies only: an existing user session in this browser is kept.
        _set_admin_auth_cookies(response, session_token, csrf_token, device_token, settings)
        return {"user": auth.user_payload(user), "csrf_token": csrf_token}

    @app.get("/api/admin/status")
    async def admin_status(request: Request) -> dict:
        """Non-error admin authentication state, derived from the admin session only."""
        raw_token = request.cookies.get(ADMIN_SESSION_COOKIE)
        if not raw_token:
            return {"authenticated": False, "reason": "anonymous", "user": None}
        try:
            session = auth.get_session(raw_token, expected_scope=ADMIN_SCOPE)
        except AuthError as exc:
            _log_auth_failure(request, ADMIN_SCOPE, exc)
            return {"authenticated": False, "reason": exc.code, "user": None}
        if session.user.role != "admin" or session.user.status != "active":
            return {"authenticated": False, "reason": "admin_forbidden", "user": None}
        return {
            "authenticated": True,
            "reason": "authenticated",
            "user": auth.user_payload(session.user),
            "session_expires_at": session.expires_at.isoformat(),
        }

    @app.post("/api/admin/logout")
    async def admin_logout(request: Request, response: Response) -> dict:
        session = _admin_csrf_session(request, auth)
        auth.logout(request.cookies.get(ADMIN_SESSION_COOKIE))
        auth_logger.info("admin logout", extra={"session_id": session.session_id, "user_id": session.user.id})
        # Clears admin cookies only; the user session in the same browser survives.
        _clear_admin_auth_cookies(response, settings)
        return {"status": "ok"}

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
            session = auth.get_session(raw_token, expected_scope=USER_SCOPE)
        except AuthError as exc:
            _log_auth_failure(request, USER_SCOPE, exc)
            return {"authenticated": False, "reason": exc.code, "user": None}
        return {
            "authenticated": True,
            "reason": "authenticated",
            "user": auth.user_payload(session.user),
            "session_expires_at": session.expires_at.isoformat(),
        }

    @app.get("/api/auth/me")
    async def me(request: Request) -> dict:
        session = _user_session(request, auth)
        return {"user": auth.user_payload(session.user), "session_expires_at": session.expires_at.isoformat()}

    @app.post("/api/auth/logout")
    async def logout(request: Request, response: Response) -> dict:
        session = _user_csrf_session(request, auth)
        auth.logout(request.cookies.get(SESSION_COOKIE))
        auth_logger.info("user logout", extra={"session_id": session.session_id, "user_id": session.user.id})
        # Clears user cookies only; the admin session in the same browser survives.
        _clear_auth_cookies(response, settings)
        return {"status": "ok"}

    @app.post("/api/auth/redeem")
    async def redeem(payload: RedeemCodeRequest, request: Request) -> dict:
        # Resolved strictly from the user session cookie + user CSRF token;
        # the admin session (if any) is never consulted here.
        session = _user_csrf_session(request, auth)
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

    @app.get("/api/edge-ml")
    async def edge_ml_status(request: Request, symbol: str | None = None, limit: int = 100) -> JSONResponse:
        """Experimental EDGE ML models: latest decisions, research results and paper trades."""
        auth.require_analysis_access(request.cookies.get(SESSION_COOKIE), expected_scope=USER_SCOPE)
        if edge_ml is None:
            payload = {"enabled": False, "models": [], "paper_trades": []}
        else:
            symbol = symbol.strip().upper()[:12] if symbol else None
            payload = {**edge_ml.status(symbol), "paper_trades": edge_ml.paper_trades(symbol, max(1, min(int(limit), 500)))}
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @app.post("/api/analyze", response_model=AnalysisResponse)
    async def analyze(payload: AnalyzeRequest, request: Request, background_tasks: BackgroundTasks) -> dict:
        _user_csrf_session(request, auth)
        auth.require_analysis_access(request.cookies.get(SESSION_COOKIE), expected_scope=USER_SCOPE)
        try:
            # provider calls (and a short wait for a free request slot) must not freeze other requests
            result, observations = await run_in_threadpool(service.analyze_with_observations, payload)
            if observations and settings.data_mode == "live":
                # Runs after the response is sent: storing setups for later outcome
                # labelling never delays the recommendation shown to the user.
                background_tasks.add_task(live_signal_recorder.record, result["symbol"], observations)
            return result
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
        # Status change + audit entry commit together or not at all.
        with database.transaction() as connection:
            row = connection.execute(
                "UPDATE subscription_codes SET status = 'revoked' WHERE id = ? AND status = 'available'",
                (code_id,),
            )
            if row.rowcount == 1:
                connection.execute(
                    """
                    INSERT INTO audit_logs(actor_user_id, action, target_type, target_id, details_json, created_at)
                    VALUES (?, 'revoke_code', 'subscription_code', ?, '{}', ?)
                    """,
                    (session.user.id, str(code_id), utc_now().isoformat()),
                )
        if row.rowcount != 1:
            raise _http_error(AuthError("code is not available", code="code_not_available", status_code=409))
        return {"status": "revoked"}

    return app


def _limit(limiter: RateLimiter, request: Request, action: str, settings: Settings) -> None:
    client_ip = request.client.host if request.client else "unknown"
    limiter.check(
        f"{action}:{client_ip}",
        limit=settings.login_rate_limit,
        window_seconds=settings.rate_limit_window_seconds,
    )


def _public_details(exc: AuthError) -> dict[str, Any]:
    return {key: value for key, value in exc.details.items() if key not in _PRIVATE_ERROR_DETAILS}


def _log_auth_failure(request: Request, area: str, exc: AuthError) -> None:
    """Record why authentication failed using non-secret identifiers only.

    Tokens, cookies and CSRF values are never logged; only the failure code,
    its reason (no/invalid/stale session, scope mismatch, CSRF failure...),
    the area and the internal session id when one was resolved.
    """
    auth_logger.warning(
        "auth request rejected",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "path": request.url.path,
            "area": area,
            "code": exc.code,
            "reason": exc.details.get("reason"),
            "session_id": exc.details.get("session_id"),
        },
    )


def _run_with_app(app: FastAPI, worker) -> None:
    """Start ``worker`` when the app starts serving and stop it on shutdown (lifespan wrapper)."""
    inner = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(asgi_app):
        worker.start()
        try:
            async with inner(asgi_app) as state:
                yield state
        finally:
            worker.stop()

    app.router.lifespan_context = lifespan


def _resolve(request: Request, area: str, resolver):
    try:
        return resolver()
    except AuthError as exc:
        _log_auth_failure(request, area, exc)
        raise _http_error(exc) from exc


def _user_session(request: Request, auth: AuthService):
    return _resolve(
        request,
        USER_SCOPE,
        lambda: auth.get_session(request.cookies.get(SESSION_COOKIE), expected_scope=USER_SCOPE),
    )


def _user_csrf_session(request: Request, auth: AuthService):
    return _resolve(
        request,
        USER_SCOPE,
        lambda: auth.verify_csrf(
            request.cookies.get(SESSION_COOKIE),
            request.headers.get("X-CSRF-Token"),
            expected_scope=USER_SCOPE,
        ),
    )


def _admin_csrf_session(request: Request, auth: AuthService):
    return _resolve(
        request,
        ADMIN_SCOPE,
        lambda: auth.verify_csrf(
            request.cookies.get(ADMIN_SESSION_COOKIE),
            request.headers.get("X-CSRF-Token"),
            expected_scope=ADMIN_SCOPE,
        ),
    )


def _admin_session(request: Request, auth: AuthService):
    raw_token = request.cookies.get(ADMIN_SESSION_COOKIE)
    if not raw_token and request.cookies.get(SESSION_COOKIE):
        # A signed-in user without an admin session is authenticated but not
        # authorized for the admin area.
        exc = AuthError(
            "admin access is required",
            code="admin_forbidden",
            status_code=403,
            details={"reason": "no_admin_session"},
        )
        _log_auth_failure(request, ADMIN_SCOPE, exc)
        raise _http_error(exc)
    session = _resolve(request, ADMIN_SCOPE, lambda: auth.get_session(raw_token, expected_scope=ADMIN_SCOPE))
    if session.user.role != "admin" or session.user.status != "active":
        exc = AuthError(
            "admin access is required",
            code="admin_forbidden",
            status_code=403,
            details={"reason": "authorization_failure", "session_id": session.session_id},
        )
        _log_auth_failure(request, ADMIN_SCOPE, exc)
        raise _http_error(exc)
    return session


def _admin_mutation_session(request: Request, auth: AuthService):
    # Use the reconciliation-aware admin session resolver for every privileged
    # mutation. This is important when the browser keeps a session created
    # before an admin account was recreated/restored with a new user id.
    # Reconciliation only ever runs on the admin cookie with an admin-scoped
    # session, so it can never turn a user session into an admin one.
    return _resolve(
        request,
        ADMIN_SCOPE,
        lambda: auth.resolve_admin_session(
            request.cookies.get(ADMIN_SESSION_COOKIE),
            request.headers.get("X-CSRF-Token"),
        ),
    )


def _check_bootstrap_csrf(request: Request, auth: AuthService) -> None:
    _resolve(
        request,
        USER_SCOPE,
        lambda: auth.verify_bootstrap_csrf(
            request.cookies.get(BOOTSTRAP_CSRF_COOKIE),
            request.headers.get("X-CSRF-Token"),
        ),
    )


def _check_admin_bootstrap_csrf(request: Request, auth: AuthService) -> None:
    header = request.headers.get("X-CSRF-Token")
    admin_cookie = request.cookies.get(ADMIN_BOOTSTRAP_CSRF_COOKIE)
    # The admin page uses its own bootstrap cookie. The shared bootstrap cookie
    # is still accepted (double-submit match required) for existing API clients.
    cookie = admin_cookie if admin_cookie and header and hmac.compare_digest(admin_cookie, header) else request.cookies.get(BOOTSTRAP_CSRF_COOKIE)
    _resolve(request, ADMIN_SCOPE, lambda: auth.verify_bootstrap_csrf(cookie, header))


def _set_auth_cookies(response: Response, session_token: str, csrf_token: str, device_token: str, settings: Settings) -> None:
    _set_cookie(response, SESSION_COOKIE, session_token, httponly=True, secure=settings.cookie_secure)
    _set_cookie(response, CSRF_COOKIE, csrf_token, httponly=False, secure=settings.cookie_secure)
    _set_cookie(response, DEVICE_COOKIE, device_token, httponly=True, secure=settings.cookie_secure, max_age=31536000)


def _set_admin_auth_cookies(response: Response, session_token: str, csrf_token: str, device_token: str, settings: Settings) -> None:
    _set_cookie(response, ADMIN_SESSION_COOKIE, session_token, httponly=True, secure=settings.cookie_secure)
    _set_cookie(response, ADMIN_CSRF_COOKIE, csrf_token, httponly=False, secure=settings.cookie_secure)
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


def _clear_admin_auth_cookies(response: Response, settings: Settings) -> None:
    for name in (ADMIN_SESSION_COOKIE, ADMIN_CSRF_COOKIE, ADMIN_BOOTSTRAP_CSRF_COOKIE):
        response.delete_cookie(name, path="/", secure=settings.cookie_secure, httponly=name == ADMIN_SESSION_COOKIE)


def _http_error(exc: AuthError) -> HTTPException:
    detail: dict[str, Any] = {"code": exc.code, "message": str(exc)}
    detail.update(_public_details(exc))
    return HTTPException(status_code=exc.status_code, detail=detail)


app = create_web_app()

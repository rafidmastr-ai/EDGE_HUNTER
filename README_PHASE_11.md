# EDGE HUNTER — Phase 11

**Phase:** Public Deployment & Security

This patch prepares the Phase 10 web/auth stack for a controlled public-edge deployment from the Mini PC.

### New

- `app/web/security.py`
- `scripts/run_production.ps1`
- `scripts/backup_database.py`
- `scripts/restore_database.py`
- `docs/PUBLIC_DEPLOYMENT_SECURITY_PHASE_11.md`
- `deploy/cloudflared/config.yml.example`
- Phase 11 unit/integration security tests

### Modified

- `app/web/app.py`
- `config/config_hunter.py`
- `app/auth/schemas.py`
- `main.py`
- `.env.example`

### Security behavior

Production uses explicit allowed hosts, explicit CORS origins, secure cookies, request-size limits, API rate limiting, security headers, HSTS, structured request logging, generic server errors, and health/readiness endpoints.

The backend should remain bound to `127.0.0.1` and be published through a trusted HTTPS edge/tunnel. SQLite is never a public service.

### Local verification

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The normal `main.py` workflow remains test-first. For production preflight + server:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_production.ps1
```

See `docs/PUBLIC_DEPLOYMENT_SECURITY_PHASE_11.md` for the Mini PC deployment, backup/recovery, and future VPS migration procedure.

**Stop after Phase 11 verification. Do not treat public internet exposure as tested until the real edge/tunnel and Windows restart procedure have been verified on the target Mini PC.**

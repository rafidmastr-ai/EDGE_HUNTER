# EDGE HUNTER — Phase 11: Public Deployment & Security

## Scope

Phase 11 prepares EDGE HUNTER for public internet access from the Mini PC and documents a VPS/cloud migration path. The implementation preserves server-side authentication/subscription enforcement and does not place SQLite on the public network.

## Production architecture

```text
Internet browser
      |
      | HTTPS
      v
Cloudflare / trusted public edge
      |
      | encrypted tunnel / reverse proxy
      v
127.0.0.1:8000  ->  FastAPI + Uvicorn
      |
      +--> local OHLC data
      +--> local SQLite database
```

Recommended Mini PC deployment is a tunnel/reverse-proxy edge such as Cloudflare Tunnel. The backend binds to `127.0.0.1` so the database and API are not directly exposed on the LAN/WAN. The public edge is responsible for HTTPS; the application additionally enables HSTS in production.

Example Cloudflare Tunnel ingress concept:

```yaml
ingress:
  - hostname: app.example.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Do not publish port 8000 directly to the internet. DNS/public routing should point to the tunnel/edge, not to the Mini PC's private database or SQLite file.

## Production environment

Use operating-system environment variables or a protected secrets manager. `.env.example` is documentation only; the application does not treat it as a secret store.

Required production controls include:

- `EDGE_HUNTER_ENV=production`
- `EDGE_HUNTER_DEBUG=false`
- `EDGE_HUNTER_API_HOST=127.0.0.1`
- `EDGE_HUNTER_ALLOWED_HOSTS=app.example.com`
- `EDGE_HUNTER_COOKIE_SECURE=true`
- `EDGE_HUNTER_DOCS_ENABLED=false`
- explicit `EDGE_HUNTER_CORS_ORIGINS` only when a separate frontend origin is genuinely required
- `EDGE_HUNTER_TRUST_PROXY_HEADERS=true` only when the app is directly behind a trusted local proxy/tunnel that sanitizes those headers

Admin credentials are never committed to source and raw passwords are never stored. Use the existing local admin bootstrap flow to create an administrator.

## HTTP security controls implemented

`app/web/security.py` adds:

- request IDs (`X-Request-ID`)
- structured JSON request/error logging
- request-body size enforcement
- process-local API rate limiting
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- restrictive `Referrer-Policy`
- restrictive `Permissions-Policy`
- Content Security Policy for the bundled same-origin UI
- HSTS in production
- `Cache-Control: no-store` for API responses

Authentication endpoints keep the stricter Phase 10 rate limit. The Phase 11 public API limiter adds another coarse layer before route execution.

The rate limiter is intentionally process-local. Production is therefore configured to one Uvicorn worker. A future multi-worker deployment should replace the store with shared infrastructure (for example Redis) before increasing worker count.

## CORS

CORS uses an explicit allow-list. Wildcard `*` is rejected by configuration because authenticated requests use cookies/credentials. Same-origin use does not need CORS configuration.

## Payload and error handling

Pydantic models continue to constrain all public request fields. Phase 11 additionally limits the raw body size before JSON parsing. Oversized requests return `413` without exposing framework internals.

Unhandled exceptions are logged server-side with a request ID and return a generic `500`. Internal paths, stack traces, database filenames and secrets are not returned to the client.

## Health and readiness

`GET /api/health` is a liveness endpoint and does not expose infrastructure details.

`GET /api/ready` checks database availability and returns `503` when the application is not ready. It does not return database paths, exception text or stack traces.

## Windows / Mini PC process supervision

Use `scripts/run_production.ps1` as the production launcher. It:

1. verifies the production environment is configured;
2. runs the project test suite before starting the server;
3. binds Uvicorn to the configured local address;
4. enables trusted local forwarded headers;
5. runs one worker to preserve the correctness of the process-local rate limiter.

For automatic startup/recovery on Windows Task Scheduler, create a task that runs:

```text
powershell.exe -ExecutionPolicy Bypass -File C:\path\to\EDGE_HUNTER\scripts\run_production.ps1
```

Recommended Task Scheduler behavior:

- trigger: At startup;
- run whether user is logged on or not;
- use a dedicated Windows account with minimum required permissions;
- restart the task after failure (for example after 1 minute);
- do not configure repeated parallel instances.

For planned restarts, stop the task/process normally, start it again, and verify `/api/health` followed by `/api/ready`.

## Backup and recovery

Use `scripts/backup_database.py` while the application may still be running. It uses SQLite's online backup API and performs an integrity check on the copy.

Example scheduled command:

```text
C:\path\to\EDGE_HUNTER\.venv\Scripts\python.exe C:\path\to\EDGE_HUNTER\scripts\backup_database.py --retention-days 14
```

Store backups outside the web root and preferably on a second disk or protected network/cloud location. Periodically test restoration, not only backup creation.

Restore procedure:

1. stop the application;
2. verify the selected backup with the restore script;
3. restore with `scripts/restore_database.py <backup-file>`;
4. start the application;
5. check `/api/health`, `/api/ready`, login, and a protected analysis request;
6. retain the automatic `.pre_restore_<timestamp>.db` file until recovery is confirmed.

## VPS/cloud migration path

The application boundary is intentionally kept independent of the public edge:

1. copy application code, configuration templates and protected environment values to the VPS;
2. copy a verified database backup or migrate the schema/data to a server database when scale requires it;
3. move the backend process from the Mini PC to `127.0.0.1` on the VPS;
4. put Nginx/Caddy/managed load balancer or the chosen tunnel at the public edge;
5. terminate HTTPS at that edge and enable HSTS at the public domain;
6. keep the database bound to private/local networking only;
7. move scheduled backups and any future worker processes to the VPS;
8. before enabling multiple backend workers, replace the in-memory rate limiter with a shared store;
9. run the full test suite and public failure-case checks before switching DNS.

SQLite can remain valid for a low-volume single-instance deployment. A later PostgreSQL migration can be introduced behind the existing repository/service boundary without changing the public API contract.

## Public failure cases covered by tests

The Phase 11 tests cover the documented public controls: secure headers, HSTS in production, trusted hosts, explicit CORS, request-size rejection, public rate limiting, hidden production API docs, safe readiness failure handling, request IDs, server-side authentication gating, and application re-initialization against the same database.

## Acceptance status

Implementation is considered ready for local verification only after the project's complete test suite passes. Public DNS/Cloudflare configuration, Windows Task Scheduler, and the actual Mini PC restart cycle remain deployment-environment actions and are not claimed as executed by the code build itself.

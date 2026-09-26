# Phase 10 — Authentication, Subscriptions & Admin

Phase 10 adds server-side account access control to the Phase 09 analysis API.

## Security model

- Passwords use PBKDF2-HMAC-SHA256 with a self-describing salt/hash record; raw passwords are never stored.
- Sessions use opaque random bearer tokens. Only SHA-256 token hashes are stored in SQLite.
- Each authenticated session has an independent CSRF token. Registration/login additionally use a bootstrap CSRF cookie/header pair.
- A browser receives one random HttpOnly device token. User accounts are bound to one device token hash. Admin relink clears the binding and revokes the user's sessions so the next login establishes a new binding.
- Sensitive public operations use a single-process sliding-window rate limiter. Phase 11 should replace it with shared infrastructure for multi-process/public deployment.
- Cookies are `HttpOnly`, `SameSite=Lax`, and `Secure` automatically in production mode.
- Admin credentials are not hard-coded. Create the first admin through `scripts/create_admin.py`.

## Public flow

1. `/api/auth/csrf` issues a bootstrap CSRF token.
2. `/api/auth/register` creates an account and a 7-day trial, then logs the user in.
3. `/api/auth/login` restores a session only from the bound device.
4. `/api/auth/me` returns the current trial/subscription access state.
5. `/api/auth/redeem` consumes a 16-character code and creates a 1- or 3-month subscription.
6. `/api/analyze` is server-side protected. It rejects missing sessions, expired sessions, disabled users and users whose trial/subscription has ended.
7. The response provides the Arabic CTA `اشترك الآن` when access is expired. The WhatsApp URL comes from `EDGE_HUNTER_WHATSAPP_URL`.

## Admin flow

Admin users use `/api/admin/login`. The admin area has its own session, CSRF token and bootstrap token, fully independent from the user area, so a user and an admin can be signed in from the same browser at the same time:

| Area  | Session cookie (HttpOnly) | Session CSRF cookie | Bootstrap CSRF (endpoint → cookie)             | Status / logout                          |
|-------|---------------------------|---------------------|------------------------------------------------|------------------------------------------|
| User  | `eh_session`              | `eh_csrf`           | `/api/auth/csrf` → `eh_bootstrap_csrf`         | `/api/auth/status`, `/api/auth/logout`   |
| Admin | `eh_admin_session`        | `eh_admin_csrf`     | `/api/admin/csrf` → `eh_admin_bootstrap_csrf`  | `/api/admin/status`, `/api/admin/logout` |

User endpoints accept only `user`-scoped sessions from `eh_session`; admin endpoints accept only `admin`-scoped sessions from `eh_admin_session`. Admin login/logout never touches the user cookies and user login/logout never touches the admin cookies. `eh_device` is shared because it identifies the browser, not a session. Admin endpoints are role protected and cover dashboard counts, users, code generation/status, revoke, extension, device relinking and audit logs.

Generate the first admin locally:

```powershell
python scripts/create_admin.py --email admin@example.com
```

The script prompts for the password and stores only its hash.

## Database

Migration version 2 adds users, trial devices, subscriptions, subscription codes, auth sessions and audit logs. The migration runner remains idempotent.

## Tests

Phase 10 tests cover registration, login/logout, 7-day trial enforcement, subscription code validation/expiry/use, one-device binding, admin authorization, revoke, extension, relink, CSRF and session expiry. The project `main.py` still runs the entire unittest suite before startup.

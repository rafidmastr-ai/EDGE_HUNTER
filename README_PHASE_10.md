# EDGE HUNTER — Phase 10

**Authentication, Subscriptions & Admin**

Phase 10 connects the public dashboard to a secure server-side account and access layer while retaining the existing Phase 09 analysis engine.

### Delivered

- Email registration/login/logout.
- Seven-day server-side free trial.
- PBKDF2-HMAC-SHA256 password hashing; no raw passwords stored.
- Opaque database-backed sessions with expiry.
- CSRF protection for authentication bootstrap and state-changing authenticated requests.
- Per-browser device binding for public user accounts, with admin-controlled relink that revokes sessions.
- In-process rate limiting for sensitive authentication/redeem endpoints.
- 16-character alphanumeric subscription codes.
- One-month and three-month subscriptions.
- Single-use code redemption with status, assigned user and expiry handling.
- Server-side analysis authorization: no valid session/access means `/api/analyze` is rejected.
- Arabic `اشترك الآن` CTA backed by `EDGE_HUNTER_WHATSAPP_URL`.
- Admin dashboard, code generation/status, user/subscription management, revoke, extend, relink and audit log.
- Secure cookie flags: HttpOnly, SameSite=Lax, and Secure in production.
- Admin bootstrap script without hard-coded credentials.

### First admin

```powershell
python scripts/create_admin.py --email admin@example.com
```

The password is entered interactively and stored only as a password hash.

### Verification

The full project suite passes before application startup. Phase 10 acceptance coverage includes registration, login/logout, trial expiry, code validation/expiry/use, device binding, admin authorization, subscription revoke/extension, session expiry, CSRF and rate limiting.

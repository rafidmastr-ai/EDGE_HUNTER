(() => {
  const el = (id) => document.getElementById(id);
  let csrfToken = null;

  const cookie = (name) => document.cookie.split('; ').find(row => row.startsWith(`${name}=`))?.slice(name.length + 1) || null;

  async function bootstrapCsrf() {
    const response = await fetch('/api/auth/csrf');
    if (!response.ok) throw new Error('تعذر تهيئة حماية الطلبات');
    const data = await response.json();
    csrfToken = data.csrf_token;
  }

  async function request(url, options = {}) {
    const headers = new Headers(options.headers || {});
    // Authenticated mutations must use the session-bound CSRF token.
    // On a page reload, csrfToken can still hold the bootstrap token,
    // which is intentionally different from the token stored with the session.
    const token = cookie('eh_csrf') || csrfToken;
    if (token) headers.set('X-CSRF-Token', token);
    if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
    return fetch(url, { ...options, headers });
  }

  async function ensureAdmin() {
    const response = await fetch('/api/auth/status');
    if (!response.ok) return false;
    const body = await response.json();
    return body.authenticated === true && body.user?.role === 'admin';
  }

  async function login(event) {
    event.preventDefault();
    try {
      if (!csrfToken) await bootstrapCsrf();
      const response = await request('/api/admin/login', {
        method: 'POST',
        body: JSON.stringify({ email: el('admin-email').value, password: el('admin-password').value }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body?.detail?.message || 'تعذر الدخول');
      csrfToken = body.csrf_token || csrfToken;
      showDashboard();
    } catch (error) {
      el('admin-login-error').hidden = false;
      el('admin-login-error').textContent = error.message;
    }
  }

  async function showDashboard() {
    el('admin-login-panel').hidden = true;
    el('admin-dashboard').hidden = false;
    await refresh();
  }

  function stat(name, value) { return `<div class="admin-stat"><span>${name}</span><strong>${value}</strong></div>`; }

  async function refresh() {
    const [dashboardResponse, usersResponse, codesResponse, auditResponse] = await Promise.all([
      request('/api/admin/dashboard'), request('/api/admin/users'), request('/api/admin/codes'), request('/api/admin/audit?limit=50'),
    ]);
    if ([dashboardResponse, usersResponse, codesResponse, auditResponse].some(r => r.status === 401 || r.status === 403)) throw new Error('انتهت جلسة الإدارة أو لم تعد الصلاحية متاحة');
    const dashboard = await dashboardResponse.json();
    const users = await usersResponse.json();
    const codes = await codesResponse.json();
    const audit = await auditResponse.json();

    el('admin-generated').textContent = new Date(dashboard.generated_at).toLocaleString('ar-IQ');
    const c = dashboard.counts;
    el('admin-stats').innerHTML = [
      stat('المستخدمون', c.users), stat('تجربة فعالة', c.trial_active), stat('اشتراك فعال', c.subscription_active),
      stat('منتهية الصلاحية', c.access_expired), stat('أكواد متاحة', c.codes_available), stat('أكواد مستخدمة', c.codes_redeemed),
    ].join('');

    el('users-body').innerHTML = users.users.map(user => {
      const expired = user.access_code === 'subscription_expired';
      const buttons = user.role === 'user' ? `<div class="admin-action-group"><button class="admin-action" data-act="extend" data-id="${user.id}">تمديد</button><button class="admin-action danger" data-act="revoke" data-id="${user.id}">إلغاء</button><button class="admin-action" data-act="relink" data-id="${user.id}">إعادة ربط الجهاز</button></div>` : '';
      return `<tr><td>${user.email}</td><td>${user.status}</td><td>${expired ? 'منتهية' : user.access_label_ar}</td><td>${user.access_expires_at ? new Date(user.access_expires_at).toLocaleDateString('ar-IQ') : '—'}</td><td>${user.device_bound ? 'مرتبط' : '—'}</td><td>${buttons}</td></tr>`;
    }).join('');

    el('codes-body').innerHTML = codes.codes.map(code => {
      const action = code.status === 'available' ? `<button class="admin-action danger" data-code-revoke="${code.id}">إلغاء</button>` : '';
      return `<tr><td>${code.code_hint}</td><td>${code.duration_months} شهر</td><td>${code.status}</td><td>${code.assigned_email || '—'}</td><td>${action}</td></tr>`;
    }).join('');
    el('audit-body').innerHTML = audit.audit.map(row => `<tr><td>${new Date(row.created_at).toLocaleString('ar-IQ')}</td><td>${row.actor_email || 'system'}</td><td>${row.action}</td><td>${row.target_type}:${row.target_id || ''}</td></tr>`).join('');
  }

  async function act(url, body = null) {
    const response = await request(url, { method: 'POST', body: body ? JSON.stringify(body) : undefined });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.detail?.message || 'تعذر تنفيذ العملية');
    await refresh();
  }

  document.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-act]');
    const revokeCode = event.target.closest('[data-code-revoke]');
    try {
      if (button) {
        const id = Number(button.dataset.id);
        const action = button.dataset.act;
        if (action === 'extend') await act(`/api/admin/users/${id}/extend`, { duration_months: 1 });
        if (action === 'revoke') await act(`/api/admin/users/${id}/revoke`, { reason: 'admin action' });
        if (action === 'relink') await act(`/api/admin/users/${id}/relink`);
      }
      if (revokeCode) await act(`/api/admin/codes/${Number(revokeCode.dataset.codeRevoke)}/revoke`);
    } catch (error) { alert(error.message); }
  });

  el('admin-login-form').addEventListener('submit', login);
  el('admin-logout').addEventListener('click', async () => {
    await request('/api/auth/logout', { method: 'POST' });
    window.location.href = '/admin';
  });
  el('code-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const response = await request('/api/admin/codes', { method: 'POST', body: JSON.stringify({ duration_months: Number(el('code-duration').value), quantity: Number(el('code-quantity').value) }) });
      const body = await response.json();
      if (!response.ok) {
        const detail = body?.detail || {};
        if (detail.code === 'admin_code_generation_failed' && detail.diagnostic_saved) {
          throw new Error(`${detail.message}\nRequest ID: ${detail.request_id || '—'}`);
        }
        throw new Error(detail.message || 'تعذر توليد الأكواد');
      }
      el('generated-codes').textContent = body.codes.map(item => `${item.code}  |  ${item.duration_months} شهر`).join('\n');
      await refresh();
    } catch (error) { alert(error.message); }
  });

  (async () => {
    try {
      await bootstrapCsrf();
      if (await ensureAdmin()) await showDashboard();
    } catch (_) {}
  })();
})();

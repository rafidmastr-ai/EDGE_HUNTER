(() => {
  const el = (id) => document.getElementById(id);
  const states = {
    initial: { label: 'جاهز', detail: 'لم يتم تشغيل التحليل بعد.', icon: '✦' },
    loading: { label: 'جاري التحليل', detail: 'يتم تحميل OHLC وتشغيل الاستراتيجيات تلقائيًا.', icon: '…' },
    success: { label: 'تحليل مكتمل', detail: 'تمت معالجة البيانات الحقيقية من واجهة API.', icon: '✓' },
    weak: { label: 'إشارة ضعيفة', detail: 'الإشارة ظاهرة، لكن قوتها محدودة وفق الدرجة الحالية.', icon: '•' },
    medium: { label: 'إشارة متوسطة', detail: 'الإشارة تحمل دعمًا متوسطًا وفق المكونات المتاحة.', icon: '•' },
    strong: { label: 'إشارة قوية', detail: 'يوجد دعم متماسك نسبيًا للقرار الحالي.', icon: '✓' },
    very_strong: { label: 'إشارة قوية جدًا', detail: 'القرار مدعوم بدرجة مرتفعة ضمن النموذج الحالي.', icon: '✓' },
    no_clear_signal: { label: 'لا توجد إشارة واضحة', detail: 'لم تتوافر أدلة كافية لاختيار اتجاه نهائي.', icon: '—' },
    api_error: { label: 'خطأ في API', detail: 'تعذر إتمام طلب التحليل.', icon: '!' },
    data_unavailable: { label: 'البيانات غير متاحة', detail: 'تعذر الحصول على بيانات السوق الحية لهذا الرمز.', icon: '!' },
    session_expired: { label: 'انتهت الجلسة', detail: 'يرجى إعادة تسجيل الدخول.', icon: '!' },
    unauthorized: { label: 'غير مصرح', detail: 'ليس لديك صلاحية لاستخدام التحليل.', icon: '!' },
    subscription_expired: { label: 'انتهى الاشتراك', detail: 'يتطلب التحليل اشتراكًا فعالًا.', icon: '!' },
  };

  const directionText = { BUY: 'BUY', SELL: 'SELL', NO_CLEAR_SIGNAL: 'NO CLEAR SIGNAL' };
  // Two different tokens, never interchangeable:
  // - bootstrapToken: pre-login double-submit token, only for /api/auth/login|register.
  // - the user session CSRF token: read from the eh_csrf cookie (set by the server
  //   together with the user session), so it stays correct after reloads and is
  //   never replaced by the admin area, which uses its own separate cookies.
  let bootstrapToken = null;
  let sessionCsrfToken = null;
  let symbolCategory = 'all';
  let symbolSearchTimer = null;
  let symbolRequestController = null;
  let authenticated = false;

  const confidenceState = (label) => {
    if (label === 'ضعيف') return 'weak';
    if (label === 'متوسط') return 'medium';
    if (label === 'قوي') return 'strong';
    if (label === 'قوي جدًا') return 'very_strong';
    return 'no_clear_signal';
  };

  function setState(state, detailOverride = null) {
    const info = states[state] || states.api_error;
    const banner = el('state-banner');
    banner.dataset.state = state;
    el('state-icon').textContent = info.icon;
    el('state-label').textContent = info.label;
    el('state-detail').textContent = detailOverride || info.detail;
  }

  function setConnection(online) {
    el('connection-status').innerHTML = `<span class="status-dot"></span><span>${online ? 'متصل' : 'غير متصل'}</span>`;
  }

  function setDataSource(label, healthy = true) {
    const node = el('data-source-status');
    node.textContent = `مصدر البيانات: ${label}`;
    node.dataset.state = healthy ? 'healthy' : 'error';
  }

  function formatPrice(value) {
    if (value === null || value === undefined) return '—';
    return Number(value).toLocaleString('en-US', { maximumFractionDigits: 6 });
  }

  function renderResult(data) {
    const state = data.status === 'no_clear_signal' ? 'no_clear_signal' : confidenceState(data.confidence_label_ar);
    const source = data.metadata?.data_source === 'twelvedata' ? 'Twelve Data' : 'البيانات المحلية';
    setDataSource(source, true);
    setState(state, data.status === 'success' ? `تمت معالجة بيانات OHLC من ${source} عبر واجهة API.` : null);
    el('result-title').textContent = data.direction === 'NO_CLEAR_SIGNAL' ? 'لا توجد إشارة واضحة' : 'نتيجة تحليل السوق';
    el('result-subtitle').textContent = `${data.symbol} — ${data.analyzed_timeframes.join(' / ')} تمت قراءتها تلقائيًا.`;
    el('analysis-time').textContent = new Date(data.timestamp).toLocaleTimeString('ar-IQ', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    el('direction').textContent = directionText[data.direction] || data.direction;
    el('direction-symbol').textContent = data.symbol;
    el('direction-card').dataset.direction = data.direction;
    el('confidence').textContent = `${Math.round(Number(data.confidence))}%`;
    el('confidence-label').textContent = data.confidence_label_ar || '—';
    el('selected-strategy').textContent = data.selected_strategy || '—';
    el('selected-variant').textContent = data.selected_variant || '—';
    el('entry').textContent = formatPrice(data.entry);
    el('target').textContent = formatPrice(data.target);
    el('stop-loss').textContent = formatPrice(data.stop_loss);
    el('lot').textContent = data.lot_size === null ? '—' : Number(data.lot_size).toFixed(2);
    el('timeframe-badge').textContent = data.primary_timeframe || 'AUTO';
    el('chart-meta').textContent = `${data.symbol} · ${data.chart.length} شمعة`;

    const reasons = el('reasons');
    reasons.innerHTML = '';
    (data.reasons || []).slice(0, 8).forEach((reason) => {
      const li = document.createElement('li');
      li.textContent = reason;
      reasons.appendChild(li);
    });
    if (!reasons.children.length) {
      const li = document.createElement('li');
      li.textContent = 'لا توجد أسباب إضافية.';
      reasons.appendChild(li);
    }

    const tbody = el('comparison-body');
    tbody.innerHTML = '';
    (data.strategy_comparison || []).forEach((row) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${row.timeframe}</td><td>${row.direction}</td><td>${Number(row.confidence).toFixed(1)}%</td><td>${row.strategy || '—'}</td>`;
      tbody.appendChild(tr);
    });
    if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="4">لا توجد نتيجة بعد.</td></tr>';

    const impact = data.capital_impact;
    const impactCard = el('capital-impact');
    if (impact) {
      impactCard.hidden = false;
      el('profit-impact').textContent = `+$${Number(impact.target_profit).toFixed(2)}`;
      el('loss-impact').textContent = `-$${Number(impact.stop_loss_loss).toFixed(2)}`;
      el('risk-impact').textContent = `$${Number(impact.risk_amount).toFixed(2)}`;
    } else {
      impactCard.hidden = true;
    }

    window.__lastChart = { candles: data.chart || [], markers: { entry: data.entry, target: data.target, stop: data.stop_loss } };
    drawChart(window.__lastChart.candles, window.__lastChart.markers);
  }

  // Clear a previous result so an error for one symbol is never shown next to
  // the prices/decision of another symbol.
  function clearResult(symbol) {
    el('result-title').textContent = 'نتيجة تحليل السوق';
    el('result-subtitle').textContent = symbol ? `${symbol} — لا توجد نتيجة.` : 'لا توجد نتيجة.';
    el('direction').textContent = '—';
    el('direction-symbol').textContent = symbol || '—';
    el('direction-card').dataset.direction = '';
    el('confidence').textContent = '—';
    el('confidence-label').textContent = '—';
    el('selected-strategy').textContent = '—';
    el('selected-variant').textContent = '—';
    ['entry', 'target', 'stop-loss', 'lot'].forEach((id) => { el(id).textContent = '—'; });
    el('comparison-body').innerHTML = '<tr><td colspan="4">لا توجد نتيجة بعد.</td></tr>';
    el('capital-impact').hidden = true;
    window.__lastChart = { candles: [], markers: {} };
    drawChart([], {});
  }

  function drawChart(candles, markers) {
    const canvas = el('chart');
    const empty = el('chart-empty');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);
    if (!candles.length) { empty.style.display = 'grid'; return; }
    empty.style.display = 'none';

    const pad = { l: 12, r: 58, t: 14, b: 22 };
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const max = Math.max(...highs, markers.entry ?? -Infinity, markers.target ?? -Infinity, markers.stop ?? -Infinity);
    const min = Math.min(...lows, markers.entry ?? Infinity, markers.target ?? Infinity, markers.stop ?? Infinity);
    const range = Math.max(max - min, 1e-9);
    const xStep = (w - pad.l - pad.r) / Math.max(candles.length - 1, 1);
    const y = (price) => pad.t + (max - price) / range * (h - pad.t - pad.b);

    ctx.strokeStyle = 'rgba(126, 160, 198, 0.11)';
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      const yy = pad.t + (h - pad.t - pad.b) * i / 5;
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    }

    const bodyWidth = Math.max(2, Math.min(9, xStep * .58));
    candles.forEach((c, i) => {
      const x = pad.l + i * xStep;
      const bullish = c.close >= c.open;
      const top = y(Math.max(c.open, c.close));
      const bottom = y(Math.min(c.open, c.close));
      ctx.strokeStyle = bullish ? '#22d88c' : '#ff4f64';
      ctx.fillStyle = bullish ? '#22d88c' : '#ff4f64';
      ctx.beginPath(); ctx.moveTo(x, y(c.high)); ctx.lineTo(x, y(c.low)); ctx.stroke();
      ctx.fillRect(x - bodyWidth / 2, top, bodyWidth, Math.max(2, bottom - top));
    });

    const markerDefs = [['entry', markers.entry, '#2bd5e6'], ['target', markers.target, '#22d88c'], ['stop', markers.stop, '#ff4f64']];
    markerDefs.forEach(([name, value, color]) => {
      if (value === null || value === undefined) return;
      const yy = y(Number(value));
      ctx.strokeStyle = color;
      ctx.setLineDash([8, 5]);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.font = '10px Segoe UI, Arial';
      ctx.textAlign = 'left';
      ctx.fillText(`${name.toUpperCase()} ${formatPrice(value)}`, w - pad.r + 6, yy + 3);
    });

    ctx.fillStyle = '#7890aa';
    ctx.font = '9px Segoe UI, Arial';
    ctx.textAlign = 'right';
    [max, min].forEach((p, idx) => ctx.fillText(formatPrice(p), w - 5, idx ? h - pad.b : pad.t + 3));
  }

  function getCookie(name) {
    const prefix = `${name}=`;
    return document.cookie.split('; ').find(row => row.startsWith(prefix))?.slice(prefix.length) || null;
  }

  async function bootstrapCsrf() {
    const response = await fetch('/api/auth/csrf');
    if (!response.ok) throw new Error('csrf bootstrap failed');
    const data = await response.json();
    bootstrapToken = data.csrf_token;
    return bootstrapToken;
  }

  function withJson(options, token) {
    const headers = new Headers(options.headers || {});
    if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
    if (token) headers.set('X-CSRF-Token', token);
    return { ...options, headers };
  }

  // Authenticated user requests: always the user session CSRF token.
  async function authFetch(url, options = {}) {
    const token = getCookie('eh_csrf') || sessionCsrfToken;
    return fetch(url, withJson(options, token));
  }

  // Pre-login requests (login/register): the bootstrap token.
  async function bootstrapFetch(url, options = {}) {
    if (!bootstrapToken) await bootstrapCsrf();
    return fetch(url, withJson(options, bootstrapToken));
  }

  function showAuthState(user) {
    authenticated = true;
    el('account-pill').textContent = user.email;
    el('auth-email').textContent = user.email;
    el('login-form').classList.add('hidden');
    el('register-form').classList.add('hidden');
    el('login-tab').classList.add('hidden');
    el('register-tab').classList.add('hidden');
    el('auth-actions').hidden = false;
    renderAccess(user);
  }

  function renderAccess(user) {
    const access = user.access || {};
    el('subscription-label').textContent = access.label_ar || '—';
    el('subscription-expiry').textContent = access.expires_at ? `ينتهي: ${new Date(access.expires_at).toLocaleDateString('ar-IQ')}` : 'لا يوجد اشتراك فعال';
    el('access-status').textContent = access.allowed ? (access.trial_active ? 'التجربة المجانية فعالة' : 'الحساب لديه صلاحية تحليل فعالة') : 'لا توجد صلاحية تحليل فعالة';
    const link = el('subscribe-link');
    if (!access.allowed) {
      link.classList.remove('hidden');
      fetch('/api/meta').then(r => r.json()).then(meta => { if (meta.whatsapp_url) link.href = meta.whatsapp_url; else link.removeAttribute('href'); }).catch(() => {});
    } else {
      link.classList.add('hidden');
    }
    el('analyze-button').disabled = !access.allowed;
  }

  async function refreshMe() {
    const response = await fetch('/api/auth/status');
    if (response.ok) {
      const data = await response.json();
      if (data.authenticated && data.user) {
        showAuthState(data.user);
        return true;
      }
    }
    authenticated = false;
    sessionCsrfToken = null;
    el('auth-email').textContent = '';
    el('account-pill').textContent = 'غير مسجل';
    el('auth-actions').hidden = true;
    el('login-tab').classList.remove('hidden');
    el('register-tab').classList.remove('hidden');
    el('login-form').classList.remove('hidden');
    el('access-status').textContent = 'سجّل الدخول للمتابعة.';
    el('analyze-button').disabled = true;
    return false;
  }

  async function submitAuth(kind) {
    const data = kind === 'login'
      ? { email: el('login-email').value, password: el('login-password').value }
      : { email: el('register-email').value, password: el('register-password').value, password_confirm: el('register-password-confirm').value };
    const response = await bootstrapFetch(`/api/auth/${kind}`, { method: 'POST', body: JSON.stringify(data) });
    const body = await response.json();
    if (!response.ok) throw new Error(body?.detail?.message || 'فشل تسجيل الدخول');
    sessionCsrfToken = body.csrf_token || null;
    showAuthState(body.user);
  }

  async function logout() {
    const response = await authFetch('/api/auth/logout', { method: 'POST' });
    if (!response.ok) {
      if (response.status === 401) { await refreshMe(); return; }
      throw new Error('فشل تسجيل الخروج');
    }
    sessionCsrfToken = null;
    authenticated = false;
    await bootstrapCsrf();
    await refreshMe();
    setState('initial', 'تم تسجيل الخروج.');
  }

  async function redeem(event) {
    event.preventDefault();
    const code = el('redeem-code').value.trim().toUpperCase();
    const response = await authFetch('/api/auth/redeem', { method: 'POST', body: JSON.stringify({ code }) });
    const body = await response.json();
    if (!response.ok) {
      // Never keep showing an account the server no longer recognises.
      if (response.status === 401) await refreshMe();
      throw new Error(body?.detail?.message || 'تعذر تفعيل الكود');
    }
    await refreshMe();
    el('redeem-code').value = '';
  }

  function normalizeSymbolInput(value) {
    return String(value || '').trim().toUpperCase().replace(/\s+/g, '');
  }

  function selectSymbol(item) {
    el('symbol').value = item.symbol;
    el('symbol-selected').textContent = `${item.symbol} — ${item.name}`;
    closeSymbolSuggestions();
  }

  function renderSymbolSuggestions(payload) {
    const box = el('symbol-suggestions');
    box.innerHTML = '';
    const results = payload.results || [];
    if (!results.length) {
      const empty = document.createElement('div');
      empty.className = 'symbol-empty';
      empty.textContent = 'لا توجد رموز مطابقة.';
      box.appendChild(empty);
      box.classList.remove('hidden');
      el('symbol').setAttribute('aria-expanded', 'true');
      return;
    }
    results.forEach((item) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'symbol-suggestion';
      button.setAttribute('role', 'option');
      const group = item.group ? ` · ${item.group}` : '';
      button.innerHTML = `<strong>${item.symbol}</strong><span>${item.name}${group}</span><em>${item.category}</em>`;
      button.addEventListener('mousedown', (event) => { event.preventDefault(); selectSymbol(item); });
      box.appendChild(button);
    });
    box.classList.remove('hidden');
    el('symbol').setAttribute('aria-expanded', 'true');
  }

  function closeSymbolSuggestions() {
    const box = el('symbol-suggestions');
    box.classList.add('hidden');
    el('symbol').setAttribute('aria-expanded', 'false');
  }

  async function searchSymbols(query = '') {
    if (symbolRequestController) symbolRequestController.abort();
    symbolRequestController = new AbortController();
    const params = new URLSearchParams({ q: query, category: symbolCategory, limit: '20' });
    try {
      const response = await fetch(`/api/symbols/search?${params.toString()}`, { signal: symbolRequestController.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.detail?.message || 'تعذر تحميل قائمة الرموز');
      renderSymbolSuggestions(data);
    } catch (error) {
      if (error.name === 'AbortError') return;
      renderSymbolSuggestions({ results: [] });
    }
  }

  function scheduleSymbolSearch() {
    clearTimeout(symbolSearchTimer);
    symbolSearchTimer = setTimeout(() => searchSymbols(normalizeSymbolInput(el('symbol').value)), 160);
  }

  function setupSymbolPicker() {
    const input = el('symbol');
    input.addEventListener('focus', () => searchSymbols(normalizeSymbolInput(input.value)));
    input.addEventListener('input', () => {
      el('symbol-clear').classList.toggle('hidden', !input.value);
      scheduleSymbolSearch();
    });
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closeSymbolSuggestions();
      if (event.key === 'Enter') {
        const first = el('symbol-suggestions').querySelector('.symbol-suggestion');
        if (first) { event.preventDefault(); first.click(); }
      }
    });
    el('symbol-clear').addEventListener('click', () => {
      input.value = '';
      input.focus();
      searchSymbols('');
    });
    document.querySelectorAll('.symbol-filter').forEach((button) => {
      button.addEventListener('click', () => {
        symbolCategory = button.dataset.category || 'all';
        document.querySelectorAll('.symbol-filter').forEach((item) => {
          const active = item === button;
          item.classList.toggle('active', active);
          item.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        searchSymbols(normalizeSymbolInput(input.value));
      });
    });
    document.addEventListener('click', (event) => {
      if (!el('symbol-picker').contains(event.target)) closeSymbolSuggestions();
    });
  }

  function riskValue() {
    const selected = el('risk-percent').value;
    return selected === 'custom' ? Number(el('custom-risk').value || 1) : Number(selected);
  }

  async function analyze() {
    if (!authenticated) { setState('unauthorized', 'يجب تسجيل الدخول أولاً.'); return; }
    const button = el('analyze-button');
    button.disabled = true;
    setState('loading');
    try {
      const response = await authFetch('/api/analyze', {
        method: 'POST',
        body: JSON.stringify({
          symbol: el('symbol').value,
          risk_percent: riskValue(),
          capital: el('capital').value ? Number(el('capital').value) : null,
          lot_mode: el('lot-mode').value,
          lot_size: Number(el('lot-size').value || 0.01),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        const code = data?.detail?.code || 'api_error';
        // Live provider failures (live_* / provider_*) are data availability problems.
        const liveDataError = code.startsWith('live_') || code.startsWith('provider_');
        const mapped = liveDataError ? 'data_unavailable' : ({ data_unavailable: 'data_unavailable', subscription_expired: 'subscription_expired', unauthorized: 'unauthorized', session_expired: 'session_expired' }[code] || 'api_error');
        clearResult(el('symbol').value);
        setState(mapped, liveDataError ? `${states.data_unavailable.detail} (${code})` : (data?.detail?.message || null));
        if (code === 'session_expired' || code === 'unauthorized') await refreshMe();
        return;
      }
      renderResult(data);
    } catch (error) {
      setState('api_error', error.message || 'تعذر الوصول إلى خدمة التحليل المحلية.');
    } finally {
      if (authenticated) {
        const meResponse = await fetch('/api/auth/status').catch(() => null);
        if (meResponse?.ok) {
          const me = await meResponse.json();
          el('analyze-button').disabled = !me.authenticated || !me.user?.access?.allowed;
        }
      }
    }
  }

  async function checkHealth() {
    try {
      const [healthResponse, providerResponse] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/live-health'),
      ]);
      setConnection(healthResponse.ok);
      if (providerResponse.ok) {
        const provider = await providerResponse.json();
        const label = provider.provider === 'twelvedata' ? 'Twelve Data LIVE' : 'Local / fallback';
        setDataSource(label, provider.status !== 'unavailable');
      } else {
        setDataSource('غير متاح', false);
      }
    } catch (_) {
      setConnection(false);
      setDataSource('غير متاح', false);
    }
  }

  function setAuthTab(mode) {
    const login = mode === 'login';
    el('login-tab').classList.toggle('active', login);
    el('register-tab').classList.toggle('active', !login);
    el('login-form').classList.toggle('hidden', !login);
    el('register-form').classList.toggle('hidden', login);
  }

  document.querySelectorAll('[data-copy]').forEach((button) => {
    button.addEventListener('click', async () => {
      const id = button.dataset.copy;
      const value = el(id).textContent;
      if (!value || value === '—') return;
      try { await navigator.clipboard.writeText(value); } catch (_) {}
      el('toast').classList.add('show');
      setTimeout(() => el('toast').classList.remove('show'), 1300);
    });
  });

  setupSymbolPicker();
  el('login-tab').addEventListener('click', () => setAuthTab('login'));
  el('register-tab').addEventListener('click', () => setAuthTab('register'));
  el('login-form').addEventListener('submit', async (event) => { event.preventDefault(); try { await submitAuth('login'); } catch (error) { setState('api_error', error.message); } });
  el('register-form').addEventListener('submit', async (event) => { event.preventDefault(); try { await submitAuth('register'); } catch (error) { setState('api_error', error.message); } });
  el('logout-button').addEventListener('click', async () => { try { await logout(); } catch (error) { setState('api_error', error.message); } });
  el('redeem-form').addEventListener('submit', async (event) => { try { await redeem(event); } catch (error) { setState('api_error', error.message); } });
  el('risk-percent').addEventListener('change', () => el('custom-risk-wrap').classList.toggle('hidden', el('risk-percent').value !== 'custom'));
  el('lot-mode').addEventListener('change', () => el('manual-lot-wrap').classList.toggle('hidden', el('lot-mode').value !== 'manual'));
  el('analyze-button').addEventListener('click', analyze);
  el('theme-toggle').addEventListener('click', () => document.body.classList.toggle('light-preview'));
  setInterval(() => {
    el('clock').textContent = new Intl.DateTimeFormat('ar-IQ', { timeZone: 'Asia/Baghdad', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }).format(new Date());
  }, 1000);
  window.addEventListener('resize', () => { if (window.__lastChart) drawChart(window.__lastChart.candles, window.__lastChart.markers); });
  window.__renderPhase10 = { setState, renderResult, drawChart, refreshMe };

  (async () => {
    checkHealth();
    setState('initial');
    try {
      await bootstrapCsrf();
      await refreshMe();
    } catch (_) {
      setState('api_error', 'تعذر تهيئة طبقة المصادقة.');
      el('analyze-button').disabled = true;
    }
  })();
})();

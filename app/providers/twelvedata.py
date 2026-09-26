"""Twelve Data live OHLC adapter.

The adapter converts the provider's response into EDGE HUNTER's canonical
``OHLCBar`` objects and keeps provider-specific parsing, throttling, retries
and caching outside the analysis/strategy layers.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.domain.market import OHLCBar
from app.providers.models import LiveProviderError, LiveProviderHealth


TIMEFRAME_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
}

SYMBOL_MAP = {
    "EURUSD": "EUR/USD",
    "XAUUSD": "XAU/USD",
}

# Quote currencies recognised when a compact symbol (e.g. ``BTCUSD``) has to be
# split into Twelve Data's ``BASE/QUOTE`` form. Longest suffixes first.
QUOTE_CURRENCIES = (
    "USDT", "USDC", "BUSD",
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SGD", "HKD",
    "NOK", "SEK", "DKK", "PLN", "TRY", "ZAR", "MXN", "CNH", "CNY",
    "BTC", "ETH",
)


def to_provider_symbol(symbol: str) -> str:
    """Return Twelve Data's ``BASE/QUOTE`` symbol for Forex, Metals and Crypto.

    ``BTC/USD`` is already provider form; ``BTCUSD``/``GBPUSD``/``ETHUSDT`` are
    split on a known quote currency. Anything else is passed through unchanged.
    """
    key = symbol.upper().strip().replace(" ", "")
    if "/" in key:
        return key
    if key in SYMBOL_MAP:
        return SYMBOL_MAP[key]
    for quote in QUOTE_CURRENCIES:
        base = key[: -len(quote)]
        if key.endswith(quote) and len(base) >= 2:
            return f"{base}/{quote}"
    return key

MAX_PROVIDER_POINTS = 5000
DEFAULT_MAX_BARS = 1000


class TwelveDataLiveProvider:
    """Configurable Twelve Data adapter using only the Python standard library."""

    name = "twelvedata"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.twelvedata.com",
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        cache_ttl_seconds: float = 15.0,
        max_bars: int = DEFAULT_MAX_BARS,
        rate_limit_per_minute: int = 8,
        transport: Callable[..., bytes] | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = max(1.0, float(timeout_seconds))
        self._max_retries = max(0, int(max_retries))
        self._backoff = max(0.0, float(backoff_seconds))
        self._cache_ttl = max(0.0, float(cache_ttl_seconds))
        self._max_bars = max(50, min(int(max_bars), MAX_PROVIDER_POINTS))
        self._rate_limit = max(1, int(rate_limit_per_minute))
        self._transport = transport or self._default_transport
        self._cache: dict[tuple[str, str], tuple[float, tuple[OHLCBar, ...]]] = {}
        self._request_times: deque[float] = deque()
        self._last_success_at: datetime | None = None
        self._last_failure_at: datetime | None = None
        self._last_latency_ms: float | None = None
        self._consecutive_failures = 0
        self._requests_total = 0
        self._cache_hits = 0
        self._last_error_code: str | None = None
        self._reference_cache: tuple[float, tuple[dict[str, str], ...]] | None = None

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def health(self) -> LiveProviderHealth:
        status = "healthy" if self.configured and self._consecutive_failures == 0 and self._last_success_at else (
            "degraded" if self.configured and self._last_success_at else
            "unavailable" if self.configured and self._last_failure_at else
            "unconfigured"
        )
        return LiveProviderHealth(
            provider=self.name,
            configured=self.configured,
            status=status,
            last_success_at=self._last_success_at,
            last_failure_at=self._last_failure_at,
            latency_ms=self._last_latency_ms,
            consecutive_failures=self._consecutive_failures,
            requests_total=self._requests_total,
            cache_hits=self._cache_hits,
            last_error_code=self._last_error_code,
        )

    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCBar]:
        if not self.configured:
            raise LiveProviderError(
                "live market data provider is not configured",
                code="live_provider_unconfigured",
                status_code=503,
            )
        if timeframe.upper() not in TIMEFRAME_MAP:
            raise LiveProviderError(
                f"unsupported live timeframe: {timeframe}",
                code="live_timeframe_unsupported",
                status_code=422,
            )
        provider_symbol = to_provider_symbol(symbol)
        symbol_key = provider_symbol
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        if start_utc > end_utc:
            raise LiveProviderError(
                "live data start must not be after end",
                code="live_invalid_range",
                status_code=422,
            )

        now = time.monotonic()
        cache_key = (symbol_key, timeframe.upper())
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] <= self._cache_ttl:
            subset = [bar for bar in cached[1] if start_utc <= bar.timestamp <= end_utc]
            if subset:
                self._cache_hits += 1
                return subset

        self._enforce_local_rate_limit()
        params = {
            "symbol": provider_symbol,
            "interval": TIMEFRAME_MAP[timeframe.upper()],
            "start_date": start_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "outputsize": str(self._max_bars),
            "timezone": "UTC",
        }
        url = f"{self._base_url}/time_series?{urlencode(params)}"
        request = Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"apikey {self._api_key}",
                "User-Agent": "EDGE-HUNTER/13",
            },
        )
        payload = self._request_json(request)
        bars = self._parse_payload(payload, start_utc, end_utc)
        if not bars:
            self._mark_failure("live_empty_data")
            raise LiveProviderError(
                "live provider returned no usable OHLC bars",
                code="live_empty_data",
                status_code=503,
            )
        self._cache[cache_key] = (time.monotonic(), tuple(bars))
        self._mark_success()
        return bars

    def get_symbol_catalog(self) -> list[dict[str, str]]:
        """Fetch the provider's daily-updated Forex, Crypto and Metal universe."""
        if not self.configured:
            raise LiveProviderError(
                "live market data provider is not configured",
                code="live_provider_unconfigured",
                status_code=503,
            )
        now = time.monotonic()
        cached = self._reference_cache
        if cached and now - cached[0] <= 86_400:
            return [dict(item) for item in cached[1]]

        collected: list[dict[str, str]] = []
        endpoints = (
            ("forex_pairs", "forex"),
            ("cryptocurrencies", "crypto"),
            ("commodities", "metals"),
        )
        for endpoint, category in endpoints:
            self._enforce_local_rate_limit()
            params = {
                "format": "JSON",
            }
            url = f"{self._base_url}/{endpoint}?{urlencode(params)}"
            request = Request(
                url,
                method="GET",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"apikey {self._api_key}",
                    "User-Agent": "EDGE-HUNTER/13",
                },
            )
            payload = self._request_json(request)
            values = payload.get("data")
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, Mapping):
                    continue
                if category == "metals" and "metal" not in str(item.get("category", "")).lower():
                    continue
                symbol = str(item.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                if category == "forex":
                    name = f"{item.get('currency_base', '')} / {item.get('currency_quote', '')}".strip(" /")
                    group = str(item.get("currency_group") or "")
                else:
                    name = str(item.get("name") or item.get("currency_base") or symbol)
                    group = str(item.get("category") or "")
                collected.append(
                    {
                        "symbol": symbol,
                        "category": category,
                        "name": name,
                        "group": group,
                        "description": str(item.get("description") or ""),
                    }
                )

        if not collected:
            raise LiveProviderError(
                "live provider returned no usable symbol catalog",
                code="live_symbol_catalog_empty",
                status_code=503,
            )

        deduped: dict[tuple[str, str], dict[str, str]] = {}
        for item in collected:
            deduped[(item["category"], item["symbol"])] = item
        ordered = tuple(
            sorted(
                deduped.values(),
                key=lambda item: ({"forex": 0, "crypto": 1, "metals": 2}[item["category"]], item["symbol"]),
            )
        )
        self._reference_cache = (time.monotonic(), ordered)
        return [dict(item) for item in ordered]

    def _request_json(self, request: Request) -> Mapping[str, Any]:
        started = time.monotonic()
        self._requests_total += 1
        last_error: LiveProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._transport(request, self._timeout)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise LiveProviderError(
                        "live provider returned an invalid JSON object",
                        code="live_invalid_response",
                    )
                if str(payload.get("status", "")).lower() == "error" or payload.get("code"):
                    code_value = str(payload.get("code") or "provider_error")
                    message = str(payload.get("message") or "live provider request failed")
                    raise LiveProviderError(
                        message,
                        code=f"provider_{code_value}",
                        status_code=429 if code_value in {"429", "rate_limit"} else 503,
                        retryable=code_value in {"429", "rate_limit"},
                    )
                self._last_latency_ms = (time.monotonic() - started) * 1000.0
                return payload
            except HTTPError as exc:
                status = int(exc.code)
                retryable = status == 429 or status >= 500
                retry_after = _retry_after_seconds(exc.headers.get("Retry-After")) if exc.headers else None
                last_error = LiveProviderError(
                    "live provider HTTP request failed",
                    code="provider_rate_limited" if status == 429 else f"provider_http_{status}",
                    status_code=429 if status == 429 else 503,
                    retryable=retryable,
                )
                if not retryable or attempt >= self._max_retries:
                    break
                time.sleep(retry_after if retry_after is not None else self._backoff * (2**attempt))
            except (TimeoutError, URLError):
                last_error = LiveProviderError(
                    "live provider network request failed",
                    code="provider_unreachable",
                    status_code=503,
                    retryable=True,
                )
                if attempt >= self._max_retries:
                    break
                time.sleep(self._backoff * (2**attempt))
            except json.JSONDecodeError:
                last_error = LiveProviderError(
                    "live provider returned malformed JSON",
                    code="live_invalid_json",
                    status_code=503,
                )
                break
            except LiveProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self._max_retries:
                    break
                time.sleep(self._backoff * (2**attempt))

        assert last_error is not None
        self._mark_failure(last_error.code)
        raise last_error

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is from controlled config
            return response.read()

    def _parse_payload(
        self,
        payload: Mapping[str, Any],
        start: datetime,
        end: datetime,
    ) -> list[OHLCBar]:
        values = payload.get("values")
        if not isinstance(values, list):
            raise LiveProviderError(
                "live provider response has no values array",
                code="live_invalid_response",
            )
        bars: dict[datetime, OHLCBar] = {}
        for item in values:
            if not isinstance(item, Mapping):
                continue
            try:
                timestamp = _parse_provider_timestamp(item.get("datetime"))
                open_price = _positive_decimal(item.get("open"))
                high_price = _positive_decimal(item.get("high"))
                low_price = _positive_decimal(item.get("low"))
                close_price = _positive_decimal(item.get("close"))
                if high_price < max(open_price, close_price, low_price):
                    continue
                if low_price > min(open_price, close_price, high_price):
                    continue
                volume = _optional_decimal(item.get("volume"))
                bars[timestamp] = OHLCBar(
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                )
            except (ValueError, InvalidOperation, TypeError):
                continue
        ordered = [bars[key] for key in sorted(bars) if start <= key <= end]
        return ordered[-self._max_bars :]

    def _enforce_local_rate_limit(self) -> None:
        now = time.monotonic()
        while self._request_times and now - self._request_times[0] >= 60.0:
            self._request_times.popleft()
        if len(self._request_times) >= self._rate_limit:
            self._mark_failure("provider_local_rate_limited")
            raise LiveProviderError(
                "local provider rate limit reached; retry later",
                code="provider_local_rate_limited",
                status_code=429,
                retryable=True,
            )
        self._request_times.append(now)

    def _mark_success(self) -> None:
        self._last_success_at = datetime.now(timezone.utc)
        self._consecutive_failures = 0
        self._last_error_code = None

    def _mark_failure(self, code: str) -> None:
        self._last_failure_at = datetime.now(timezone.utc)
        self._consecutive_failures += 1
        self._last_error_code = code


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_provider_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty provider timestamp")
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return _as_utc(parsed)


def _positive_decimal(value: Any) -> Decimal:
    number = Decimal(str(value).strip())
    if not number.is_finite() or number <= 0:
        raise ValueError("invalid OHLC value")
    return number


def _optional_decimal(value: Any) -> Decimal | None:
    if value in (None, "", "null"):
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    if not number.is_finite() or number < 0:
        return None
    return number


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())

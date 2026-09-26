"""Web-facing market symbol catalog backed by the configured live provider.

The catalog is intentionally separate from local CSV datasets used by
backtests/learning. In production the catalog is refreshed from provider
reference data and cached in-process for a day to avoid repeated upstream
requests while still reflecting the provider's daily-updated symbol lists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Iterable

from app.providers.live_provider import LiveMarketDataProvider
from app.providers.models import LiveProviderError


CATEGORY_ALL = "all"
CATEGORY_FOREX = "forex"
CATEGORY_CRYPTO = "crypto"
CATEGORY_METALS = "metals"
VALID_CATEGORIES = {CATEGORY_ALL, CATEGORY_FOREX, CATEGORY_CRYPTO, CATEGORY_METALS}


class SymbolCatalogUnavailableError(RuntimeError):
    """Raised when a configured live provider cannot supply its symbol catalog."""

    def __init__(self, message: str, *, code: str = "symbol_catalog_unavailable") -> None:
        super().__init__(message)
        self.code = code

POPULAR_SYMBOLS = (
    "XAU/USD",
    "XAG/USD",
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD",
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
)


@dataclass(frozen=True)
class SymbolItem:
    symbol: str
    code: str
    name: str
    category: str
    group: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def compact_symbol(symbol: str) -> str:
    """Normalize provider-style slash-delimited symbols for matching/display."""
    return "".join(ch for ch in str(symbol).upper().strip() if ch.isalnum() or ch in "._:-")


class SymbolCatalogService:
    """Cache, normalize and search the provider's supported market universe."""

    def __init__(
        self,
        provider: LiveMarketDataProvider,
        *,
        cache_ttl_seconds: int = 86_400,
        max_results: int = 20,
    ) -> None:
        self.provider = provider
        self.cache_ttl_seconds = max(300, int(cache_ttl_seconds))
        self.max_results = max(5, min(int(max_results), 50))
        self._cache: tuple[float, tuple[SymbolItem, ...]] | None = None
        self._lock = Lock()

    def search(self, query: str = "", category: str = CATEGORY_ALL, limit: int | None = None) -> dict:
        category = str(category or CATEGORY_ALL).strip().lower()
        if category not in VALID_CATEGORIES:
            raise ValueError(f"unsupported symbol category: {category}")
        requested_limit = self.max_results if limit is None else max(1, min(int(limit), self.max_results))
        items, cache_source = self._get_catalog()
        normalized_query = compact_symbol(query)

        candidates = [item for item in items if category == CATEGORY_ALL or item.category == category]
        if normalized_query:
            ranked = sorted(candidates, key=lambda item: self._score(item, normalized_query))
            results = [item for item in ranked if self._matches(item, normalized_query)][:requested_limit]
        else:
            results = self._popular(candidates)[:requested_limit]

        return {
            "query": str(query or "").strip(),
            "category": category,
            "count": len(results),
            "results": [item.to_dict() for item in results],
            "source": cache_source,
            "cached": self._cache is not None,
        }

    def _get_catalog(self) -> tuple[tuple[SymbolItem, ...], str]:
        now = datetime.now(timezone.utc).timestamp()
        cached = self._cache
        if cached and now - cached[0] <= self.cache_ttl_seconds:
            return cached[1], "twelvedata"

        with self._lock:
            cached = self._cache
            if cached and now - cached[0] <= self.cache_ttl_seconds:
                return cached[1], "twelvedata"
            try:
                raw_items = self.provider.get_symbol_catalog()
            except LiveProviderError as exc:
                try:
                    configured = bool(self.provider.health().configured)
                except Exception:
                    configured = True
                if not configured:
                    fallback = self._fallback_catalog()
                    return fallback, "fallback"
                raise SymbolCatalogUnavailableError(
                    "live market symbol catalog is currently unavailable",
                    code=exc.code,
                ) from exc

            normalized = self._normalize_items(raw_items)
            if not normalized:
                fallback = self._fallback_catalog()
                return fallback, "fallback"
            self._cache = (now, tuple(normalized))
            return tuple(normalized), "twelvedata"

    @staticmethod
    def _normalize_items(raw_items: Iterable[dict]) -> list[SymbolItem]:
        seen: set[tuple[str, str]] = set()
        normalized: list[SymbolItem] = []
        category_rank = {CATEGORY_FOREX: 0, CATEGORY_CRYPTO: 1, CATEGORY_METALS: 2}
        for raw in raw_items:
            symbol = str(raw.get("symbol", "")).strip().upper()
            category = str(raw.get("category", "")).strip().lower()
            if category not in VALID_CATEGORIES - {CATEGORY_ALL}:
                continue
            compact = compact_symbol(symbol)
            if len(compact) < 4:
                continue
            key = (category, compact)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                SymbolItem(
                    symbol=symbol,
                    code=compact,
                    name=str(raw.get("name") or raw.get("currency_base") or symbol).strip(),
                    category=category,
                    group=str(raw.get("group") or raw.get("currency_group") or "").strip(),
                    description=str(raw.get("description") or "").strip(),
                )
            )
        normalized.sort(key=lambda item: (category_rank[item.category], item.symbol))
        return normalized

    @staticmethod
    def _matches(item: SymbolItem, query: str) -> bool:
        fields = (
            compact_symbol(item.symbol),
            compact_symbol(item.code),
            compact_symbol(item.name),
            compact_symbol(item.description),
            compact_symbol(item.group),
        )
        return any(query in field for field in fields)

    @staticmethod
    def _score(item: SymbolItem, query: str) -> tuple[int, int, str]:
        code = compact_symbol(item.code)
        name = compact_symbol(item.name)
        if code == query:
            rank = 0
        elif code.startswith(query):
            rank = 1
        elif compact_symbol(item.symbol).startswith(query):
            rank = 2
        elif name.startswith(query):
            rank = 3
        else:
            rank = 4
        return rank, len(code), item.symbol

    @staticmethod
    def _popular(items: list[SymbolItem]) -> list[SymbolItem]:
        by_code = {item.code: item for item in items}
        popular = [by_code[compact_symbol(symbol)] for symbol in POPULAR_SYMBOLS if compact_symbol(symbol) in by_code]
        remaining = [item for item in items if item not in popular]
        return popular + remaining

    @staticmethod
    def _fallback_catalog() -> tuple[SymbolItem, ...]:
        fallback = (
            ("EUR/USD", "Euro / US Dollar", CATEGORY_FOREX, "Major"),
            ("GBP/USD", "British Pound / US Dollar", CATEGORY_FOREX, "Major"),
            ("USD/JPY", "US Dollar / Japanese Yen", CATEGORY_FOREX, "Major"),
            ("USD/CHF", "US Dollar / Swiss Franc", CATEGORY_FOREX, "Major"),
            ("AUD/USD", "Australian Dollar / US Dollar", CATEGORY_FOREX, "Major"),
            ("USD/CAD", "US Dollar / Canadian Dollar", CATEGORY_FOREX, "Major"),
            ("NZD/USD", "New Zealand Dollar / US Dollar", CATEGORY_FOREX, "Major"),
            ("EUR/GBP", "Euro / British Pound", CATEGORY_FOREX, "Minor"),
            ("XAU/USD", "Gold Spot", CATEGORY_METALS, "Precious Metal"),
            ("XAG/USD", "Silver Spot", CATEGORY_METALS, "Precious Metal"),
            ("XPT/USD", "Platinum Spot", CATEGORY_METALS, "Precious Metal"),
            ("XPD/USD", "Palladium Spot", CATEGORY_METALS, "Precious Metal"),
            ("BTC/USD", "Bitcoin / US Dollar", CATEGORY_CRYPTO, ""),
            ("ETH/USD", "Ethereum / US Dollar", CATEGORY_CRYPTO, ""),
            ("SOL/USD", "Solana / US Dollar", CATEGORY_CRYPTO, ""),
        )
        return tuple(
            SymbolItem(symbol=symbol, code=compact_symbol(symbol), name=name, category=category, group=group)
            for symbol, name, category, group in fallback
        )

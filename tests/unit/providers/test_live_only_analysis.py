"""Live Analysis must use the live provider (Twelve Data) only, never local CSV."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from app.db.database import Database
from app.domain.market import OHLCBar
from app.providers.models import LiveProviderError, LiveProviderHealth
from app.providers.twelvedata import TwelveDataLiveProvider, to_provider_symbol
from app.web.analysis_service import (
    LiveDataUnavailableError,
    LocalOHLCAnalysisService,
    _contract_size_for_symbol,
    _floor_timeframe,
)
from app.web.app import create_web_app
from app.web.schemas import AnalyzeRequest
from config.config_hunter import load_settings

INTERVAL_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240}
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}


def _series(end: datetime, minutes: int, count: int, start_price: float) -> list[tuple[datetime, Decimal, Decimal, Decimal, Decimal]]:
    first = end - timedelta(minutes=minutes * (count - 1))
    rows = []
    price = Decimal(str(start_price))
    step = Decimal(str(round(start_price * 0.0004, 6)))
    for i in range(count):
        ts = first + timedelta(minutes=minutes * i)
        drift = step if (i // 25) % 2 == 0 else -step * Decimal("0.6")
        close = price + drift
        rows.append((ts, price, max(price, close) + step, min(price, close) - step, close))
        price = close
    return rows


class FakeTwelveDataTransport:
    """Stands in for the HTTPS call to api.twelvedata.com and records requests."""

    def __init__(self, *, error: dict | None = None, prices: dict[str, float] | None = None) -> None:
        self.error = error
        self.prices = prices or {"BTC/USD": 65000.0, "ETH/USD": 3200.0, "XAU/USD": 2400.0, "GBP/USD": 1.27}
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self.error is not None:
            return json.dumps(self.error).encode("utf-8")
        query = {key: values[0] for key, values in parse_qs(urlparse(request.full_url).query).items()}
        minutes = INTERVAL_MINUTES[query["interval"]]
        end = datetime.strptime(query["end_date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        end = _floor_timeframe(end, minutes)
        rows = _series(end, minutes, 420, self.prices.get(query["symbol"], 100.0))
        values = [
            {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": str(o),
                "high": str(h),
                "low": str(low),
                "close": str(c),
            }
            for ts, o, h, low, c in rows
        ]
        return json.dumps({"status": "ok", "values": list(reversed(values))}).encode("utf-8")


class RecordingProvider:
    name = "mock-live"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def get_ohlc(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[OHLCBar]:
        self.calls.append((symbol, timeframe))
        if self.fail:
            raise LiveProviderError("provider down", code="provider_unreachable", status_code=503)
        minutes = TIMEFRAME_MINUTES[timeframe]
        rows = _series(_floor_timeframe(end, minutes) - timedelta(minutes=minutes), minutes, 420, 65000.0)
        return [OHLCBar(timestamp=ts, open=o, high=h, low=low, close=c, volume=None) for ts, o, h, low, c in rows]

    def health(self) -> LiveProviderHealth:
        return LiveProviderHealth(provider=self.name, configured=True, status="healthy")


def _write_csv(path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("timestamp,open,high,low,close\n")
        for i in range(600):
            handle.write(f"{(start + timedelta(minutes=i)).isoformat()},100,101,99,100.5\n")


class LiveOnlyAnalysisServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # CSVs exist for XAUUSD (backtest data) but must never be touched live.
        _write_csv(self.root / "XAUUSD.csv")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _service(self, provider) -> LocalOHLCAnalysisService:
        service = LocalOHLCAnalysisService(data_root=self.root, live_provider=provider, data_mode="live", live_history_bars=420)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("live analysis must not read local CSV data")

        service._load_dataset = forbidden  # type: ignore[method-assign]
        service._read_csv = forbidden  # type: ignore[method-assign]
        return service

    def test_crypto_symbols_are_analysed_from_live_data_without_csv(self) -> None:
        for symbol in ("BTC/USD", "ETH/USD"):
            provider = RecordingProvider()
            result = self._service(provider).analyze(AnalyzeRequest(symbol=symbol, risk_percent=1.0, capital=1000.0))
            self.assertEqual(result["symbol"], symbol)
            self.assertEqual(result["metadata"]["data_source"], "mock-live")
            self.assertEqual(result["metadata"]["data_mode"], "live")
            self.assertFalse(result["metadata"]["data_fallback_used"])
            self.assertEqual(result["analyzed_timeframes"], ["M5", "M15", "H1"])
            self.assertEqual(provider.calls, [(symbol, "M5"), (symbol, "M15"), (symbol, "H1")])

    def test_provider_failure_is_reported_and_never_falls_back_to_csv(self) -> None:
        provider = RecordingProvider(fail=True)
        with self.assertRaises(LiveDataUnavailableError) as ctx:
            self._service(provider).analyze(AnalyzeRequest(symbol="XAUUSD"))
        self.assertEqual(ctx.exception.code, "provider_unreachable")
        self.assertNotIn("CSV", str(ctx.exception))

    def test_missing_provider_is_a_live_error_not_a_csv_error(self) -> None:
        with self.assertRaises(LiveDataUnavailableError) as ctx:
            self._service(None).analyze(AnalyzeRequest(symbol="BTC/USD"))
        self.assertEqual(ctx.exception.code, "live_provider_unconfigured")

    def test_forming_candle_is_excluded_from_live_bars(self) -> None:
        now = datetime(2026, 9, 26, 12, 7, 30, tzinfo=timezone.utc)
        self.assertEqual(_floor_timeframe(now, 5), datetime(2026, 9, 26, 12, 5, tzinfo=timezone.utc))
        self.assertEqual(_floor_timeframe(now, 60), datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc))

    def test_contract_sizes_cover_forex_metals_and_crypto(self) -> None:
        self.assertEqual(_contract_size_for_symbol("EUR/USD"), 100_000.0)
        self.assertEqual(_contract_size_for_symbol("GBPUSD"), 100_000.0)
        self.assertEqual(_contract_size_for_symbol("XAU/USD"), 100.0)
        self.assertEqual(_contract_size_for_symbol("XAGUSD"), 5000.0)
        self.assertEqual(_contract_size_for_symbol("BTC/USD"), 1.0)
        self.assertEqual(_contract_size_for_symbol("ETH/USD"), 1.0)

    def test_compact_symbols_map_to_twelve_data_form(self) -> None:
        cases = {
            "BTC/USD": "BTC/USD",
            "BTCUSD": "BTC/USD",
            "ETHUSD": "ETH/USD",
            "ETHUSDT": "ETH/USDT",
            "GBPUSD": "GBP/USD",
            "XAUUSD": "XAU/USD",
            "EURUSD": "EUR/USD",
            "USDJPY": "USD/JPY",
        }
        for symbol, expected in cases.items():
            self.assertEqual(to_provider_symbol(symbol), expected, symbol)


class LiveAnalysisApiTests(unittest.TestCase):
    """End-to-end: /api/analyze → TwelveDataLiveProvider → (fake) Twelve Data HTTPS."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_root = root / "raw"
        self.data_root.mkdir()
        _write_csv(self.data_root / "XAUUSD.csv")  # backtest data only
        self.settings = replace(
            load_settings(),
            environment="test",
            database_path=root / "edge.db",
            log_dir=root / "logs",
            password_iterations=100_000,
            cookie_secure=False,
            data_mode="live",
            live_provider_name="twelvedata",
            live_provider_enabled=True,
            live_provider_api_key="test-key-not-real",
            live_provider_url="https://api.twelvedata.com",
            live_provider_max_retries=0,
            live_provider_cache_ttl_seconds=0.0,
            live_provider_max_bars=420,
            live_provider_rate_limit_per_minute=50,
            strategy_ml_model_path=root / "models" / "strategy_learning.json",
        )
        self.database = Database(self.settings.database_path)
        self.app = create_web_app(self.data_root, database=self.database, settings=self.settings)
        self.provider = self.app.state.live_provider
        self.assertIsInstance(self.provider, TwelveDataLiveProvider)
        self.client = TestClient(self.app)
        boot = self.client.get("/api/auth/csrf").json()["csrf_token"]
        register = self.client.post(
            "/api/auth/register",
            headers={"X-CSRF-Token": boot},
            json={"email": "live@example.com", "password": "StrongPass!123", "password_confirm": "StrongPass!123"},
        )
        self.assertEqual(register.status_code, 200, register.text)
        self.csrf = register.json()["csrf_token"]

    def tearDown(self) -> None:
        self.database.close()
        self.tmp.cleanup()

    def _analyze(self, symbol: str, **extra):
        return self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": self.csrf},
            json={"symbol": symbol, "risk_percent": 1.0, "lot_mode": "auto", **extra},
        )

    def test_crypto_forex_and_metal_analysis_request_twelve_data(self) -> None:
        for symbol, provider_symbol in (("BTC/USD", "BTC/USD"), ("ETH/USD", "ETH/USD"), ("GBPUSD", "GBP/USD"), ("XAU/USD", "XAU/USD")):
            transport = FakeTwelveDataTransport()
            self.provider._transport = transport
            response = self._analyze(symbol, capital=1000)
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["metadata"]["data_source"], "twelvedata")
            self.assertEqual(body["metadata"]["data_mode"], "live")
            self.assertFalse(body["metadata"]["data_fallback_used"])
            self.assertTrue(body["chart"])
            urls = [urlparse(request.full_url) for request in transport.requests]
            self.assertEqual({url.netloc for url in urls}, {"api.twelvedata.com"})
            self.assertEqual({url.path for url in urls}, {"/time_series"})
            queries = [parse_qs(url.query) for url in urls]
            self.assertEqual({q["symbol"][0] for q in queries}, {provider_symbol})
            self.assertEqual([q["interval"][0] for q in queries], ["5min", "15min", "1h"])
            for request in transport.requests:
                self.assertEqual(request.headers["Authorization"], "apikey test-key-not-real")
                self.assertNotIn("test-key-not-real", request.full_url)

    def test_provider_error_returns_controlled_503_without_csv_fallback(self) -> None:
        self.provider._transport = FakeTwelveDataTransport(error={"status": "error", "code": 401, "message": "bad key"})
        response = self._analyze("XAUUSD")  # a local XAUUSD.csv exists and must be ignored
        self.assertEqual(response.status_code, 503, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "provider_401")
        self.assertNotIn("CSV", detail["message"])
        health = self.client.get("/api/live-health").json()
        self.assertFalse(health["fallback_to_local"])
        self.assertNotIn("test-key-not-real", json.dumps(health))

    def test_strategy_setups_are_recorded_for_learning_after_the_response(self) -> None:
        from app.signals.engine import SignalConfidenceEngine
        from app.strategies.models import SignalDirection, SignalState, StrategySignal
        from app.strategies.registry import StrategyRegistry

        class AlwaysBuy:
            name = "Classic"
            variant = "Classic_V1"

            def generate(self, context):
                close = float(context.bars[-1].close)
                return StrategySignal(
                    timestamp=context.current.timestamp, symbol=context.symbol, timeframe=context.timeframe,
                    direction=SignalDirection.BUY, state=SignalState.SIGNAL, entry=close, stop_loss=close * 0.99,
                    target=close * 1.0175, risk_reward=1.75, entry_logic="t", invalidation="t", stop_loss_logic="t",
                    target_logic="t", evidence=("always",), strategy_name=self.name, variant=self.variant,
                )

        service = self.app.state.analysis_service
        service.signal_engine = SignalConfidenceEngine(StrategyRegistry((AlwaysBuy(),)), signal_filter=service.signal_filter)
        self.provider._transport = FakeTwelveDataTransport()
        response = self._analyze("BTC/USD")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("strategy_learning", response.json()["metadata"])
        rows = self.database.execute(
            "SELECT symbol, timeframe, strategy_name, source_type, status FROM learning_records ORDER BY timeframe"
        ).fetchall()
        self.assertEqual([tuple(row) for row in rows], [
            ("BTC/USD", "H1", "Classic", "LIVE_TRADE", "PENDING_OUTCOME"),
            ("BTC/USD", "M15", "Classic", "LIVE_TRADE", "PENDING_OUTCOME"),
            ("BTC/USD", "M5", "Classic", "LIVE_TRADE", "PENDING_OUTCOME"),
        ])
        # Re-analysing the same closed candles does not duplicate the records.
        self._analyze("BTC/USD")
        self.assertEqual(self.database.execute("SELECT COUNT(*) FROM learning_records").fetchone()[0], 3)

    def test_btc_without_any_local_csv_is_not_a_csv_error(self) -> None:
        self.provider._transport = FakeTwelveDataTransport()
        response = self._analyze("BTC/USD")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("no local CSV", response.text)


if __name__ == "__main__":
    unittest.main()

# EDGE HUNTER — Phase 13

Phase 13 is the final implementation phase for the live market-data boundary. The concrete adapter currently implemented is Twelve Data, while the application remains provider-neutral through the `LiveMarketDataProvider` interface.

Run the normal project checks with:

```powershell
python main.py
```

To explicitly smoke-test the configured live provider (this consumes provider quota):

```powershell
python scripts/check_live_provider.py --symbol XAUUSD --timeframe M5 --bars 50
```

Do not run the smoke test until the provider API key and applicable license are configured.

See `docs/LIVE_DATA_PROVIDER_PHASE_13.md` for provider capabilities, licensing considerations, runtime configuration and replacement instructions.

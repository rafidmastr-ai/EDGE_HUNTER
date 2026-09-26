# Phase 03 — Analysis / Feature Engine

Phase 03 adds the reusable OHLC analysis layer required before strategy implementation.

## Run the project checks

`main.py` automatically discovers and executes the entire `unittest` suite. It stops startup if any test fails.

```powershell
python main.py
```

For detailed test output:

```powershell
python -m unittest discover -s tests -v
```

## Package

```text
app/features/
├── __init__.py
├── calculations.py
├── engine.py
├── models.py
└── multitimeframe.py
```

The implementation is strategy-independent and consumes canonical OHLC data; provider-specific access remains outside the feature calculation layer.

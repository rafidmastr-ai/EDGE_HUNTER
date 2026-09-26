# PHASE 08 — SIGNAL SELECTION & CONFIDENCE ENGINE

This phase adds the transparent signal aggregation and confidence layer.

### New package

`app/signals/`

### Main entry point

```python
SignalConfidenceEngine(StrategyRegistry.default()).analyze(context)
```

### Guarantees

- deterministic confidence;
- identical inputs → identical output;
- BUY/SELL selection requires configurable evidence gates;
- weak signals remain visible;
- conflicts and insufficient data are represented;
- no future OHLC access inside the aggregation layer;
- one final target only;
- structured `to_dict()` output for a future API;
- optional validated evidence from Phase 07;
- calibration and monotonicity diagnostics.

### Tests

Phase 08 tests are stored under:

```text
tests/unit/signals/
tests/integration/signals/
```

`main.py` runs the full project test suite, so Phase 08 tests are included
in the same automated check.

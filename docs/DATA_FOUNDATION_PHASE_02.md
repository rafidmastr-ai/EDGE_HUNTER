# EDGE HUNTER — Phase 02 Data Foundation

## Pipeline

```text
CSV discovery
    ↓
CSV loader
    ↓
column/timestamp normalization
    ↓
canonical OHLC
    ↓
deterministic sorting
    ↓
deterministic timestamp deduplication
    ↓
OHLC validation
    ↓
quality report
    ↓
clean repository slice
```

## Timestamp policy

- Source timestamps are parsed with timezone information when present.
- Naive timestamps are treated as UTC for this phase.
- Zoned timestamps are converted to UTC.
- Source text is not rewritten in-place.

## Validation

The validator checks:
- required OHLC fields
- numeric values
- positive finite prices
- monotonic timestamps
- duplicate timestamps
- OHLC consistency
- non-negative volume when volume exists

## Incomplete data

This phase does not invent or forward-fill market prices.
Missing/incomplete records are reported rather than fabricated.

Gap detection remains dependent on a known timeframe calendar and will be extended where the dataset/timeframe metadata is sufficient.

## Provider architecture

The canonical output is independent of CSV formatting. Future live providers must map into the same market-data boundary.

## Provider boundary

The canonical market-data boundary is OHLC. Provider-specific acquisition remains outside the feature/data-foundation contracts.

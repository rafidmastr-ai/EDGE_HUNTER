# EDGE HUNTER — Phase 06

**Backtest Research & Strategy Comparison**

Phase 06 turns the Phase 05 measurement instrument into a reproducible research workflow for Classic, SMC and ICT.

The implementation covers the research matrix, configurable R:R and parameter variants, chronological IS/OOS separation, consistent metrics, signal-frequency diagnostics, evidence flags, candidate extraction for Phase 07, JSON/JSONL/CSV persistence and a human-readable Markdown report.

Run the full development test suite through `main.py`. To run an actual historical research job later:

```powershell
python scripts/run_phase06_research.py --data-dir data/raw --output-dir reports/research/phase06
```

For a smaller smoke matrix:

```powershell
python scripts/run_phase06_research.py --quick
```

No configuration is called “best” in this phase, and no optimization is performed. Phase 07 consumes the explicitly recorded research candidates.

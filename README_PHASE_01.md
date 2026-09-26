# EDGE HUNTER — Phase 01
## System Architecture & Project Foundation

This phase establishes the project foundation only.

### Scope
- Modular application structure.
- Separation between domain logic and API/UI concerns.
- Central configuration through `config_hunter.py`.
- SQLite database abstraction and migration foundation.
- Historical CSV provider interface and future live-provider interface.
- Safe local development bootstrap.
- Initial import/startup tests.
- Architecture documentation.

### Explicit exclusions
- No strategy implementation.
- No backtesting implementation.
- No feature/indicator implementation.
- No live market-data provider implementation.
- No real authentication/subscription implementation yet.

### Run

From the project root:

```powershell
python main.py
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

The project intentionally uses Python standard-library components in this foundation phase so the architecture can be inspected before adding framework-specific dependencies.

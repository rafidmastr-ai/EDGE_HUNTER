# Phase 09 — httpx2 TestClient warning fix

Starlette's TestClient now prefers the `httpx2` package. The project previously relied on `httpx`, which produced a deprecation warning during the test suite.

Changed files:
- requirements-web.txt
- pyproject.toml

After replacing these files, install/update the dependency inside the project's .venv:

    python -m pip install -r requirements-web.txt

Or, for the project package dependencies:

    python -m pip install -e .

Then run:

    python main.py

The expected result is the same passing test count with no Starlette httpx deprecation warning, assuming the environment has the updated dependencies installed.

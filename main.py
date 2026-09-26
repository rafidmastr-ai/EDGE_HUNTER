"""EDGE HUNTER application entry point with separate check and serve modes."""

from __future__ import annotations

import argparse
import os

from app.core.bootstrap import create_application
from app.core.test_runner import run_project_tests


def main() -> int:
    """Run tests first; optionally start the web server after Phase 12 checks."""
    parser = argparse.ArgumentParser(description="EDGE HUNTER")
    parser.add_argument("--serve", action="store_true", help="start the web server")
    parser.add_argument("--check", action="store_true", help="run the full automated test suite and do not start the server")
    parser.add_argument("--production", action="store_true", help="serve using production proxy/header settings")
    args = parser.parse_args()

    if args.production:
        os.environ["EDGE_HUNTER_ENV"] = "production"
        os.environ["EDGE_HUNTER_DEBUG"] = "false"
        os.environ["EDGE_HUNTER_COOKIE_SECURE"] = "true"
        os.environ["EDGE_HUNTER_DOCS_ENABLED"] = "false"

    if args.check:
        return run_project_tests()

    application = create_application()
    application.start()

    print()
    print("EDGE HUNTER application initialized successfully.")
    print(f"Environment: {application.settings.environment}")
    print(f"Database: {application.settings.database_path}")
    print("RESULT: APPLICATION STARTUP PASS")

    if args.serve:
        import uvicorn

        print(f"WEB SERVER: http://{application.settings.api_host}:{application.settings.api_port}/")
        uvicorn.run(
            "app.web.app:app",
            host=application.settings.api_host,
            port=application.settings.api_port,
            reload=False,
            proxy_headers=args.production or application.settings.trust_proxy_headers,
            forwarded_allow_ips="127.0.0.1" if (args.production or application.settings.trust_proxy_headers) else "",
            server_header=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

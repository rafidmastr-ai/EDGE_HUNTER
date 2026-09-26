"""Start the local EDGE HUNTER Phase 09 web server."""

from __future__ import annotations

import uvicorn

from config.config_hunter import load_settings


if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(
        "app.web.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )

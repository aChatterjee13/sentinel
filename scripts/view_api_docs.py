#!/usr/bin/env python3
"""Launch the Sentinel API docs (Swagger UI + ReDoc) standalone.

Usage:
    python scripts/view_api_docs.py [--port 8502]

Opens:
    http://localhost:8502/docs      — Swagger UI (interactive)
    http://localhost:8502/redoc     — ReDoc (read-only reference)
    http://localhost:8502/openapi.json — Raw OpenAPI spec
"""

import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import SentinelClient
from sentinel.dashboard.server import create_dashboard_app


def main() -> None:
    parser = argparse.ArgumentParser(description="View Sentinel API documentation")
    parser.add_argument("--port", type=int, default=8502, help="Port (default: 8502)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # Minimal client — no real model needed, just enough to boot the app
    from sentinel.config.schema import SentinelConfig

    cfg = SentinelConfig(model={"name": "sentinel_api_docs", "type": "classification"})
    client = SentinelClient(cfg)
    app = create_dashboard_app(client)

    url = f"http://{args.host}:{args.port}/docs"
    print("\n  Sentinel API Documentation")
    print("  ─────────────────────────────────────")
    print(f"  Swagger UI : http://{args.host}:{args.port}/docs")
    print(f"  ReDoc      : http://{args.host}:{args.port}/redoc")
    print(f"  OpenAPI    : http://{args.host}:{args.port}/openapi.json")
    print("  ─────────────────────────────────────\n")

    if not args.no_browser:
        Timer(1.5, webbrowser.open, args=[url]).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

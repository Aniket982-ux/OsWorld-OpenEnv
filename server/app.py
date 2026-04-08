# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Osworld Environment.

This module creates an HTTP server that exposes the OsworldEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""
from dotenv import load_dotenv
import os

# Definitively enable the web interface for this deployment
os.environ["ENABLE_WEB_INTERFACE"] = "true"
load_dotenv(override=True)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import OsworldAction, OsworldObservation
    from .OsWorld_environment import OsworldEnvironment
except (ModuleNotFoundError, ImportError):
    from models import OsworldAction, OsworldObservation
    from server.OsWorld_environment import OsworldEnvironment


# Create the app with web interface and README integration
app = create_app(
    OsworldEnvironment,
    OsworldAction,
    OsworldObservation,
    env_name="OsWorld",
    max_concurrent_envs=16,  # increased to allow 16 concurrent training sessions/workers
)


def main():
    """
    Entry point for direct execution via uv run or python -m.
    """
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Run the Osworld Environment Server")
    parser.add_id = True # Dummy to prevent issues if needed
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port number to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

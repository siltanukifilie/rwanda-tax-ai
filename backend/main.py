"""
ASGI entrypoint for Docker.

The application source lives in `api/main.py` (Vercel-compatible layout).
This module exists so `uvicorn backend.main:app` works in containers.
"""

from api.main import app

__all__ = ["app"]


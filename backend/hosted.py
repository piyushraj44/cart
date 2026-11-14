"""Single-origin entry point for container hosting.

Build the Vite frontend into frontend/dist, then run
`uvicorn backend.hosted:app --host 0.0.0.0 --port 8000`.
The ordinary backend.main:app entry point remains available for local Vite dev.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.main import app

DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
app.mount("/assets", StaticFiles(directory=DIST / "assets", check_dir=False), name="frontend-assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    if path.startswith("api/") or path == "health":
        raise HTTPException(404, "Not found")
    # Vite's hashed assets live under /assets; every application route uses
    # the SPA shell. No arbitrary path from the request is read from disk.
    index = DIST / "index.html"
    if not index.is_file():
        raise HTTPException(503, "Frontend build is not available")
    return FileResponse(index)

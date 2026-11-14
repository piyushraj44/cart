"""Runs the frontend rendering tests (frontend/tests/render.test.jsx) from pytest so one command covers both."""
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


@pytest.mark.skipif(shutil.which("node") is None or not (FRONTEND / "node_modules").exists(),
                    reason="Node.js or frontend/node_modules missing (run `npm install` in frontend/)")
def test_frontend_render_suite():
    r = subprocess.run(["node", "tests/run-render-tests.mjs"], cwd=FRONTEND, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert " 0 failed" in r.stdout

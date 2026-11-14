import ml  # noqa: F401  (must be first: sets OMP_NUM_THREADS before torch/xgboost)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: live-server tests (not run by default)")

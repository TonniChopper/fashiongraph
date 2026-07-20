"""Pytest bootstrap: ensure the repo root is importable as `fg`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

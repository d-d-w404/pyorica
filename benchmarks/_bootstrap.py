"""Put repo root on ``sys.path`` when benchmark scripts are run directly."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_on_path() -> Path:
    """Insert pyorica repo root into ``sys.path`` if missing."""
    root = Path(__file__).resolve().parent.parent
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root

#!/usr/bin/env python3
"""Repo-root shim so that ``python run.py --help`` works from a fresh clone.

The real CLI lives in :mod:`dtropd.run`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtropd.run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _single_thread():
    """Keep tests single-threaded and deterministic."""
    import torch

    torch.set_num_threads(1)
    yield

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triton_flash_course.utils import is_gpu_ready  # noqa: E402


@pytest.fixture
def require_gpu():
    if not is_gpu_ready():
        pytest.skip("requires CUDA/ROCm GPU and Triton")

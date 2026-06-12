from __future__ import annotations

import pytest
import torch

from triton_flash_course.kernels.memory_ops import copy_1d, strided_read, zero_1d_like
from triton_flash_course.utils import is_gpu_ready

pytestmark = pytest.mark.gpu


def test_copy_1d_gpu() -> None:
    if not is_gpu_ready():
        pytest.skip("requires GPU + Triton")
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    y = copy_1d(x)
    torch.testing.assert_close(y, x)


def test_zero_1d_like_gpu() -> None:
    if not is_gpu_ready():
        pytest.skip("requires GPU + Triton")
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    y = zero_1d_like(x)
    torch.testing.assert_close(y, torch.zeros_like(x))


def test_strided_read_gpu() -> None:
    if not is_gpu_ready():
        pytest.skip("requires GPU + Triton")
    x = torch.randn(8192, device="cuda", dtype=torch.float32)
    y = strided_read(x, stride=4)
    torch.testing.assert_close(y, x[::4][: y.numel()])

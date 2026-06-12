from __future__ import annotations

import pytest
import torch

from triton_flash_course.kernels.blockptr_ops import copy_2d_block_ptr
from triton_flash_course.utils import is_gpu_ready

pytestmark = pytest.mark.gpu


def test_copy_2d_block_ptr_gpu() -> None:
    if not is_gpu_ready():
        pytest.skip("requires GPU + Triton")
    x = torch.randn(65, 129, device="cuda", dtype=torch.float32)
    y = copy_2d_block_ptr(x, block_m=32, block_n=64)
    torch.testing.assert_close(y, x)

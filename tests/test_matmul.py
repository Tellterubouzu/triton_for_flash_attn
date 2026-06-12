from __future__ import annotations

import pytest
import torch

from triton_flash_course.kernels.matmul import matmul
from triton_flash_course.reference import matmul_ref
from triton_flash_course.utils import assert_close


@pytest.mark.gpu
def test_matmul_gpu(require_gpu):
    a = torch.randn(64, 96, device="cuda", dtype=torch.float16)
    b = torch.randn(96, 80, device="cuda", dtype=torch.float16)
    out = matmul(a, b, block_m=16, block_n=16, block_k=32)
    ref = matmul_ref(a, b).float()
    assert_close("matmul", out, ref, dtype=torch.float16)

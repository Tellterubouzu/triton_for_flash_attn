from __future__ import annotations

import pytest
import torch

from triton_flash_course.kernels.layernorm import layernorm
from triton_flash_course.reference import layernorm_ref
from triton_flash_course.utils import assert_close


@pytest.mark.gpu
def test_layernorm_gpu(require_gpu):
    x = torch.randn(32, 257, device="cuda", dtype=torch.float32)
    w = torch.randn(257, device="cuda", dtype=torch.float32)
    b = torch.randn(257, device="cuda", dtype=torch.float32)
    out = layernorm(x, w, b)
    ref = layernorm_ref(x, w, b)
    assert_close("layernorm", out, ref, dtype=torch.float32)

from __future__ import annotations

import pytest
import torch

from triton_flash_course.kernels.fused_ops import bias_gelu, swish
from triton_flash_course.reference import swish_ref
from triton_flash_course.utils import assert_close


@pytest.mark.gpu
def test_swish_gpu(require_gpu):
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    out = swish(x)
    ref = swish_ref(x)
    assert_close("swish", out, ref, dtype=torch.float32)


@pytest.mark.gpu
def test_bias_gelu_gpu(require_gpu):
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    b = torch.randn(128, device="cuda", dtype=torch.float32)
    out = bias_gelu(x, b)
    ref = torch.nn.functional.gelu(x + b, approximate="tanh")
    assert_close("bias_gelu", out, ref, dtype=torch.float32)

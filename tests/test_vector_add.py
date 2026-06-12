from __future__ import annotations

import pytest
import torch

from triton_flash_course.kernels.vector_add import vector_add
from triton_flash_course.reference import vector_add_ref
from triton_flash_course.utils import assert_close


@pytest.mark.gpu
def test_vector_add_gpu(require_gpu):
    x = torch.randn(100_003, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    out = vector_add(x, y)
    ref = vector_add_ref(x, y)
    assert_close("vector_add", out, ref, dtype=torch.float32)

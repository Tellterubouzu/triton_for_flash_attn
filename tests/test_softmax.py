from __future__ import annotations

import pytest
import torch

from triton_flash_course.kernels.softmax import row_softmax
from triton_flash_course.reference import softmax_ref
from triton_flash_course.utils import assert_close


@pytest.mark.gpu
def test_row_softmax_gpu(require_gpu):
    x = torch.randn(64, 257, device="cuda", dtype=torch.float32)
    out = row_softmax(x)
    ref = softmax_ref(x)
    assert_close("row_softmax", out, ref, dtype=torch.float32)

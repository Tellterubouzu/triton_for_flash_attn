from __future__ import annotations

import pytest
import torch

from triton_flash_course.kernels.flash_attention import FlashConfig, flash_attention_forward
from triton_flash_course.reference import attention_ref
from triton_flash_course.utils import assert_close, make_qkv


@pytest.mark.gpu
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_forward_gpu(require_gpu, causal: bool):
    q, k, v = make_qkv(1, 2, 64, 32, dtype=torch.float16, device="cuda")
    cfg = FlashConfig(block_m=16, block_n=32)
    out = flash_attention_forward(q, k, v, causal=causal, config=cfg)
    ref = attention_ref(q, k, v, causal=causal)
    assert_close("flash_attention_forward", out, ref, dtype=torch.float16)

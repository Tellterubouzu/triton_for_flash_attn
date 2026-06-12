from __future__ import annotations

import torch

from triton_flash_course.reference import attention_ref, layernorm_ref, softmax_ref


def test_softmax_ref_rows_sum_to_one():
    x = torch.randn(8, 16)
    y = softmax_ref(x)
    torch.testing.assert_close(y.sum(dim=-1), torch.ones(8))


def test_layernorm_ref_shape():
    x = torch.randn(4, 32)
    w = torch.ones(32)
    b = torch.zeros(32)
    y = layernorm_ref(x, w, b)
    assert y.shape == x.shape


def test_attention_ref_shape():
    q = torch.randn(2, 3, 8, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    y = attention_ref(q, k, v, causal=True)
    assert y.shape == q.shape

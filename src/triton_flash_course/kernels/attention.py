from __future__ import annotations

import math

import torch

from triton_flash_course.reference import attention_ref, attention_torch_sdpa


def attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    return attention_ref(q, k, v, causal=causal)


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    return attention_torch_sdpa(q, k, v, causal=causal, sm_scale=scale)

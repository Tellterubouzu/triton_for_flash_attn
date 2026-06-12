from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def vector_add_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """c_i = a_i + b_i."""
    return a + b


def swish_ref(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """y_i = x_i * sigmoid(beta * x_i)."""
    return x * torch.sigmoid(beta * x)


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """Stable row-wise softmax for tensors of shape [M, N]."""
    x_max = x.max(dim=-1, keepdim=True).values
    z = x - x_max
    numerator = torch.exp(z)
    return numerator / numerator.sum(dim=-1, keepdim=True)


def matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C_ij = sum_k A_ik B_kj."""
    return a @ b


def layernorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return F.layer_norm(x, normalized_shape=(x.shape[-1],), weight=weight, bias=bias, eps=eps)


def attention_scores(q: torch.Tensor, k: torch.Tensor, sm_scale: float | None = None) -> torch.Tensor:
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.matmul(q, k.transpose(-2, -1)) * sm_scale


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    return_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference scaled dot-product attention.

    Shapes are [B, H, N_CTX, D_HEAD]. This implementation explicitly materializes
    the [B, H, N_CTX, N_CTX] score and probability matrices, so it is intentionally
    memory hungry. That makes it useful as a profiler baseline.
    """
    scores = attention_scores(q, k, sm_scale)
    if causal:
        n_q, n_k = q.shape[-2], k.shape[-2]
        row = torch.arange(n_q, device=q.device)[:, None]
        col = torch.arange(n_k, device=q.device)[None, :]
        scores = scores.masked_fill(col > row, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    out = torch.matmul(probs, v)
    if return_probs:
        return out, probs
    return out


def attention_torch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=sm_scale)


def estimate_attention_bytes(batch: int, heads: int, seq: int, dim: int, dtype: torch.dtype) -> dict[str, int]:
    element_size = torch.tensor([], dtype=dtype).element_size()
    qkv_out = 4 * batch * heads * seq * dim * element_size
    scores_probs = 2 * batch * heads * seq * seq * element_size
    return {
        "qkv_out_only": qkv_out,
        "materialized_scores_probs": scores_probs,
        "naive_total_lower_bound": qkv_out + scores_probs,
    }


def estimate_attention_flops(batch: int, heads: int, seq: int, dim: int) -> int:
    # Two matmuls: QK^T and P V. Softmax FLOPs are shape dependent and excluded here.
    return 4 * batch * heads * seq * seq * dim

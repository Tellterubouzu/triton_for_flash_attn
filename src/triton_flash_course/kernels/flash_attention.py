from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from triton_flash_course.reference import attention_ref
from triton_flash_course.utils import ceil_div, next_power_of_2

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


@dataclass(frozen=True)
class FlashConfig:
    block_m: int = 16
    block_n: int = 64
    num_warps: int = 4
    num_stages: int = 3


def default_flash_config(head_dim: int, seq_len: int) -> FlashConfig:
    """Small, conservative config set for teaching and portability."""
    if head_dim <= 32:
        return FlashConfig(block_m=32, block_n=64, num_warps=4, num_stages=3)
    if head_dim <= 64:
        return FlashConfig(block_m=16 if seq_len >= 2048 else 32, block_n=64, num_warps=4, num_stages=3)
    return FlashConfig(block_m=16, block_n=32, num_warps=4, num_stages=3)


if triton is not None:

    @triton.jit
    def _flash_attention_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sm_scale,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qn: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_on: tl.constexpr,
        stride_od: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        N_CTX: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        P_DTYPE: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_bh = tl.program_id(axis=1)
        b = pid_bh // H
        h = pid_bh - b * H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_base = q_ptr + b * stride_qb + h * stride_qh
        k_base = k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        o_base = o_ptr + b * stride_ob + h * stride_oh

        q = tl.load(
            q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
            mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for start_n in range(0, N_CTX, BLOCK_N):
            k_idx = start_n + offs_n
            k = tl.load(
                k_base + k_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=(k_idx[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
                other=0.0,
            )
            v = tl.load(
                v_base + k_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=(k_idx[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
                other=0.0,
            )

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(k_idx[None, :] < N_CTX, qk, -float("inf"))
            if CAUSAL:
                qk = tl.where(k_idx[None, :] <= offs_m[:, None], qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_new = l_i * alpha + tl.sum(p, axis=1)

            if P_DTYPE == 0:
                p_cast = p.to(tl.float16)
            else:
                p_cast = p.to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(p_cast, v)
            m_i = m_new
            l_i = l_new

        acc = acc / l_i[:, None]
        o_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD)
        tl.store(o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od, acc, mask=o_mask)
        tl.store(lse_ptr + pid_bh * N_CTX + offs_m, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    config: FlashConfig | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Educational FlashAttention forward kernel.

    Input shape: [batch, heads, seq, head_dim]. Only fp16/bf16 inputs are accepted.
    The kernel is intentionally simple: no dropout, no variable length, no GQA/MQA,
    no paged KV cache, and no backward pass.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("flash_attention_forward requires CUDA/ROCm tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"expected q, k, v with same shape, got {q.shape}, {k.shape}, {v.shape}")
    if q.ndim != 4:
        raise ValueError("expected q, k, v with shape [batch, heads, seq, head_dim]")
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("educational FlashAttention kernel supports fp16/bf16 inputs")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have the same dtype")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, heads, seq, head_dim = q.shape
    if head_dim > 128:
        raise ValueError("head_dim > 128 is intentionally left as an exercise")
    block_d = next_power_of_2(head_dim)
    config = config or default_flash_config(head_dim, seq)
    sm_scale = (1.0 / math.sqrt(head_dim)) if sm_scale is None else sm_scale

    o = torch.empty_like(q)
    lse = torch.empty((batch, heads, seq), device=q.device, dtype=torch.float32)
    grid = (ceil_div(seq, config.block_m), batch * heads)
    p_dtype = 0 if q.dtype == torch.float16 else 1
    _flash_attention_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B=batch,
        H=heads,
        N_CTX=seq,
        D_HEAD=head_dim,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        P_DTYPE=p_dtype,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    if return_lse:
        return o, lse
    return o


def flash_attention_or_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Convenience function for lessons that should run on CPU-only machines."""
    if triton is None or not q.is_cuda or q.dtype not in {torch.float16, torch.bfloat16}:
        return attention_ref(q, k, v, causal=causal)
    return flash_attention_forward(q, k, v, causal=causal)

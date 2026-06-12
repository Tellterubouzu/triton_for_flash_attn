from __future__ import annotations

import torch

from triton_flash_course.utils import next_power_of_2

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _layernorm_fwd_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n_cols
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * rstd * w + b
        tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise RuntimeError("layernorm Triton kernel requires CUDA/ROCm tensors")
    if x.ndim < 2:
        raise ValueError("x must have shape [..., n_cols]")
    if weight.shape != (x.shape[-1],) or bias.shape != (x.shape[-1],):
        raise ValueError("weight and bias must have shape [x.shape[-1]]")
    x2d = x.contiguous().view(-1, x.shape[-1])
    weight = weight.contiguous()
    bias = bias.contiguous()
    n_rows, n_cols = x2d.shape
    block_n = next_power_of_2(n_cols) if block_n is None else block_n
    if block_n < n_cols:
        raise ValueError("block_n must be >= n_cols")
    if num_warps is None:
        num_warps = 4 if block_n <= 1024 else 8
    y = torch.empty_like(x2d)
    _layernorm_fwd_kernel[(n_rows,)](
        x2d,
        weight,
        bias,
        y,
        n_cols,
        eps,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return y.view_as(x)

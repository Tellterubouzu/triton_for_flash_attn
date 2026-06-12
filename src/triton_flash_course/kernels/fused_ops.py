from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _swish_kernel(x_ptr, y_ptr, n_elements, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x / (1.0 + tl.exp(-beta * x))
        tl.store(y_ptr + offsets, y, mask=mask)

    @triton.jit
    def _bias_gelu_kernel(x_ptr, bias_ptr, y_ptr, n_elements, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = offsets % n_cols
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        z = x + b
        # tanh approximation used by many GELU implementations.
        y = 0.5 * z * (1.0 + tl.tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)))
        tl.store(y_ptr + offsets, y, mask=mask)


def swish(x: torch.Tensor, *, beta: float = 1.0, block_size: int = 1024) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise RuntimeError("swish Triton kernel requires CUDA/ROCm tensor")
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _swish_kernel[grid](x, y, n_elements, beta=beta, BLOCK_SIZE=block_size)
    return y


def bias_gelu(x: torch.Tensor, bias: torch.Tensor, *, block_size: int = 1024) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not (x.is_cuda and bias.is_cuda):
        raise RuntimeError("bias_gelu Triton kernel requires CUDA/ROCm tensors")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if bias.numel() != x.shape[-1]:
        raise ValueError("bias length must equal x.shape[-1]")
    x = x.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    n_cols = x.shape[-1]
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _bias_gelu_kernel[grid](x, bias, y, n_elements, n_cols, BLOCK_SIZE=block_size)
    return y

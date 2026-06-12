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
    def _row_softmax_kernel(x_ptr, y_ptr, n_rows, n_cols: tl.constexpr, BLOCK_N: tl.constexpr):
        row = tl.program_id(axis=0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        numerator = tl.exp(x)
        denominator = tl.sum(numerator, axis=0)
        y = numerator / denominator
        tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=["n_cols"],
    )
    @triton.jit
    def _row_softmax_autotuned_kernel(x_ptr, y_ptr, n_rows, n_cols: tl.constexpr, BLOCK_N: tl.constexpr):
        row = tl.program_id(axis=0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        numerator = tl.exp(x)
        denominator = tl.sum(numerator, axis=0)
        y = numerator / denominator
        tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


def row_softmax(x: torch.Tensor, *, block_n: int | None = None, num_warps: int | None = None) -> torch.Tensor:
    """Stable softmax over the last dimension of a 2D tensor."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise RuntimeError("row_softmax Triton kernel requires CUDA/ROCm tensor")
    if x.ndim != 2:
        raise ValueError("row_softmax expects a 2D tensor [n_rows, n_cols]")
    x = x.contiguous()
    n_rows, n_cols = x.shape
    block_n = next_power_of_2(n_cols) if block_n is None else block_n
    if block_n < n_cols:
        raise ValueError("block_n must be >= n_cols")
    if num_warps is None:
        num_warps = 4 if block_n <= 2048 else 8
    y = torch.empty_like(x)
    _row_softmax_kernel[(n_rows,)](x, y, n_rows, n_cols, BLOCK_N=block_n, num_warps=num_warps)
    return y



def row_softmax_autotuned(x: torch.Tensor, *, block_n: int | None = None) -> torch.Tensor:
    """Same as row_softmax, but lets Triton benchmark num_warps candidates.

    Set TRITON_PRINT_AUTOTUNING=1 to see the selected config.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise RuntimeError("row_softmax_autotuned Triton kernel requires CUDA/ROCm tensor")
    if x.ndim != 2:
        raise ValueError("row_softmax_autotuned expects a 2D tensor [n_rows, n_cols]")
    x = x.contiguous()
    n_rows, n_cols = x.shape
    block_n = next_power_of_2(n_cols) if block_n is None else block_n
    if block_n < n_cols:
        raise ValueError("block_n must be >= n_cols")
    y = torch.empty_like(x)
    _row_softmax_autotuned_kernel[(n_rows,)](x, y, n_rows, n_cols, BLOCK_N=block_n)
    return y

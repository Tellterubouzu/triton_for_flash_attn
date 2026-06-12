from __future__ import annotations

import torch

try:  # imported lazily so the repository can be inspected on CPU-only machines
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on machines without Triton
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor, *, block_size: int = 1024) -> torch.Tensor:
    """Compute c_i = a_i + b_i with one Triton program per block."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not (a.is_cuda and b.is_cuda):
        raise RuntimeError("vector_add Triton kernel requires CUDA/ROCm tensors")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape=} {b.shape=}")
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty_like(a)
    n_elements = c.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=block_size)
    return c

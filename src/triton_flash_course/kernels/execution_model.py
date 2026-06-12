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
    def _program_id_trace_kernel(pid_out, offset_out, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(pid_out + offsets, pid, mask=mask)
        tl.store(offset_out + offsets, offsets, mask=mask)

    @triton.jit
    def _tiny_dot_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def program_id_trace(n_elements: int, block_size: int = 8, device: str | torch.device = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if n_elements < 0:
        raise ValueError("n_elements must be non-negative")
    pid = torch.empty((n_elements,), device=device, dtype=torch.int64)
    offsets = torch.empty((n_elements,), device=device, dtype=torch.int64)
    grid = (triton.cdiv(n_elements, block_size),)
    _program_id_trace_kernel[grid](pid, offsets, n_elements, BLOCK_SIZE=block_size)
    return pid, offsets


def compile_tiny_dot(
    *,
    M: int = 64,
    N: int = 64,
    K: int = 64,
    dtype: torch.dtype = torch.float16,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    input_precision: str = "tf32",
    device: str | torch.device = "cuda",
):
    """Compile and run a minimal tl.dot kernel, returning (output, compiled_kernel)."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    compiled = _tiny_dot_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        INPUT_PRECISION=input_precision,
        num_warps=num_warps,
        num_stages=4,
    )
    return c, compiled

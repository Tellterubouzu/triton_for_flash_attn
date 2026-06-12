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
    def _copy_2d_block_ptr_kernel(
        x_ptr,
        y_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N
        x_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(offs_m, offs_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        y_block = tl.make_block_ptr(
            base=y_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(offs_m, offs_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
        tl.store(y_block, x, boundary_check=(0, 1))


    @triton.jit
    def _stream_kv_like_block_ptr_kernel(
        k_ptr,
        out_ptr,
        N_CTX: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # Educational kernel: one program streams K blocks along the sequence axis and
        # writes the sum over sequence for each D tile. It demonstrates tl.advance,
        # not an optimized reduction kernel.
        pid_d = tl.program_id(0)
        d_start = pid_d * BLOCK_D
        k_block = tl.make_block_ptr(
            base=k_ptr,
            shape=(N_CTX, D_HEAD),
            strides=(D_HEAD, 1),
            offsets=(0, d_start),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        # We keep a BLOCK_N x BLOCK_D accumulator only to expose block pointer
        # movement. A real reduction would reduce over BLOCK_N each iteration.
        for _ in range(0, N_CTX, BLOCK_N):
            x = tl.load(k_block, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            acc += x
            k_block = tl.advance(k_block, (BLOCK_N, 0))
        cols = d_start + tl.arange(0, BLOCK_D)
        col_mask = cols < D_HEAD
        reduced = tl.sum(acc, axis=0)
        tl.store(out_ptr + cols, reduced, mask=col_mask)


def copy_2d_block_ptr(x: torch.Tensor, *, block_m: int = 32, block_n: int = 64) -> torch.Tensor:
    """Copy a contiguous 2D tensor using tl.make_block_ptr.

    This is intentionally equivalent to y = x.clone(), but the address expression
    is moved into a block pointer descriptor with shape/strides/offsets.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise RuntimeError("copy_2d_block_ptr requires CUDA/ROCm tensor")
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    x = x.contiguous()
    y = torch.empty_like(x)
    M, N = x.shape
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _copy_2d_block_ptr_kernel[grid](x, y, M, N, BLOCK_M=block_m, BLOCK_N=block_n)
    return y


def stream_kv_like_sum(k: torch.Tensor, *, block_n: int = 64, block_d: int = 64) -> torch.Tensor:
    """Demonstrate tl.advance over the sequence axis for a [N_CTX, D_HEAD] tensor.

    Returns a toy per-column accumulation. It is not optimized and is only meant
    to make block-pointer movement concrete before reading FlashAttention.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not k.is_cuda:
        raise RuntimeError("stream_kv_like_sum requires CUDA/ROCm tensor")
    if k.ndim != 2:
        raise ValueError("k must be [N_CTX, D_HEAD]")
    k = k.contiguous()
    n_ctx, d_head = k.shape
    out = torch.empty((d_head,), dtype=torch.float32, device=k.device)
    grid = lambda meta: (triton.cdiv(d_head, meta["BLOCK_D"]),)
    _stream_kv_like_block_ptr_kernel[grid](
        k,
        out,
        n_ctx,
        d_head,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return out

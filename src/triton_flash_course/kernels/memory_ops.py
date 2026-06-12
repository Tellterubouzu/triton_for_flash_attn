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
    def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, x, mask=mask)


    @triton.jit
    def _copy_with_cache_hint_kernel(
        src_ptr,
        dst_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        LOAD_CACHE: tl.constexpr,
        STORE_CACHE: tl.constexpr,
        EVICTION: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(
            src_ptr + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=LOAD_CACHE,
            eviction_policy=EVICTION,
        )
        tl.store(dst_ptr + offsets, x, mask=mask, cache_modifier=STORE_CACHE)


    @triton.jit
    def _zero_kernel(dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        z = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        tl.store(dst_ptr + offsets, z, mask=mask)


    @triton.jit
    def _fill_kernel(dst_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        v = tl.full((BLOCK_SIZE,), value, dtype=tl.float32)
        tl.store(dst_ptr + offsets, v, mask=mask)


    @triton.jit
    def _strided_read_kernel(src_ptr, dst_ptr, n_outputs, stride: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_outputs
        x = tl.load(src_ptr + offsets * stride, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, x, mask=mask)


    @triton.jit
    def _address_map_2d_kernel(
        base_ptr,
        out_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        stride_row: tl.constexpr,
        stride_col: tl.constexpr,
        storage_offset: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        linear = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = linear < rows * cols
        r = linear // cols
        c = linear - r * cols
        offsets = storage_offset + r * stride_row + c * stride_col
        x = tl.load(base_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + linear, x, mask=mask)


def _require_triton_cuda(name: str, *tensors: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError(f"{name} requires CUDA/ROCm tensors")


def copy_1d(src: torch.Tensor, *, block_size: int = 1024) -> torch.Tensor:
    """Copy a contiguous tensor with a Triton load/store kernel.

    Bytes moved for bandwidth accounting is approximately read + write:
    2 * src.numel() * src.element_size().
    """
    _require_triton_cuda("copy_1d", src)
    src = src.contiguous()
    dst = torch.empty_like(src)
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _copy_kernel[grid](src, dst, n, BLOCK_SIZE=block_size)
    return dst


def copy_1d_with_cache_hint(
    src: torch.Tensor,
    *,
    block_size: int = 1024,
    load_cache: str = "",
    store_cache: str = "",
    eviction: str = "",
) -> torch.Tensor:
    """Copy with Triton cache hints.

    NVIDIA PTX-oriented hints:
    - load_cache: "", ".ca", ".cg", ".cv"
    - store_cache: "", ".wb", ".cg", ".cs", ".wt"
    - eviction: "", "evict_first", "evict_last"

    On non-NVIDIA backends, support is compiler-version dependent; benchmark and
    inspect generated IR/ISA instead of assuming the hint is honored.
    """
    _require_triton_cuda("copy_1d_with_cache_hint", src)
    valid_load = {"", ".ca", ".cg", ".cv"}
    valid_store = {"", ".wb", ".cg", ".cs", ".wt"}
    valid_eviction = {"", "evict_first", "evict_last"}
    if load_cache not in valid_load:
        raise ValueError(f"invalid load_cache={load_cache!r}")
    if store_cache not in valid_store:
        raise ValueError(f"invalid store_cache={store_cache!r}")
    if eviction not in valid_eviction:
        raise ValueError(f"invalid eviction={eviction!r}")
    src = src.contiguous()
    dst = torch.empty_like(src)
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _copy_with_cache_hint_kernel[grid](
        src,
        dst,
        n,
        BLOCK_SIZE=block_size,
        LOAD_CACHE=load_cache,
        STORE_CACHE=store_cache,
        EVICTION=eviction,
    )
    return dst


def zero_1d_like(x: torch.Tensor, *, block_size: int = 1024) -> torch.Tensor:
    _require_triton_cuda("zero_1d_like", x)
    dst = torch.empty_like(x.contiguous())
    n = dst.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _zero_kernel[grid](dst, n, BLOCK_SIZE=block_size)
    return dst


def fill_1d_like(x: torch.Tensor, value: float, *, block_size: int = 1024) -> torch.Tensor:
    _require_triton_cuda("fill_1d_like", x)
    dst = torch.empty_like(x.contiguous())
    n = dst.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _fill_kernel[grid](dst, n, value, BLOCK_SIZE=block_size)
    return dst


def strided_read(src: torch.Tensor, *, stride: int, n_outputs: int | None = None, block_size: int = 1024) -> torch.Tensor:
    """Read src[0], src[stride], src[2*stride], ... into a compact output.

    This isolates coalescing and cache-line waste. For stride=1, neighboring lanes
    read neighboring elements. For large stride, each lane may touch a different
    cache line/sector, lowering effective bandwidth.
    """
    _require_triton_cuda("strided_read", src)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    src = src.contiguous()
    max_outputs = (src.numel() + stride - 1) // stride
    n = max_outputs if n_outputs is None else min(n_outputs, max_outputs)
    dst = torch.empty((n,), dtype=src.dtype, device=src.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _strided_read_kernel[grid](src, dst, n, stride, BLOCK_SIZE=block_size)
    return dst


def materialize_2d_view_by_address(
    base_storage_view: torch.Tensor,
    *,
    rows: int,
    cols: int,
    stride_row: int,
    stride_col: int,
    storage_offset: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """Educational kernel that reconstructs a 2D strided view from address math.

    base_storage_view must be a 1D contiguous tensor that exposes the underlying
    storage. PyTorch does not let us safely pass a raw storage object to Triton,
    so this is a simplified exercise tool rather than a general view kernel.
    """
    _require_triton_cuda("materialize_2d_view_by_address", base_storage_view)
    if base_storage_view.ndim != 1 or not base_storage_view.is_contiguous():
        raise ValueError("base_storage_view must be a contiguous 1D tensor")
    out = torch.empty((rows, cols), dtype=base_storage_view.dtype, device=base_storage_view.device)
    n = rows * cols
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _address_map_2d_kernel[grid](
        base_storage_view,
        out,
        rows,
        cols,
        stride_row,
        stride_col,
        storage_offset,
        BLOCK_SIZE=block_size,
    )
    return out

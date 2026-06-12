from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

import torch

from triton_flash_course.kernels.vector_add import vector_add
from triton_flash_course.reference import vector_add_ref
from triton_flash_course.utils import benchmark_ms, gbps, get_device, is_gpu_ready, parse_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16_000_000)
    parser.add_argument("--dtype", type=str, default="fp32")
    args = parser.parse_args()
    device = get_device()
    dtype = parse_dtype(args.dtype)
    x = torch.randn(args.n, device=device, dtype=dtype)
    y = torch.randn_like(x)
    bytes_moved = 3 * args.n * x.element_size()
    torch_ms = benchmark_ms(lambda: vector_add_ref(x, y))
    print(f"torch: {torch_ms:.4f} ms, {gbps(bytes_moved, torch_ms):.1f} GB/s")
    if is_gpu_ready():
        triton_ms = benchmark_ms(lambda: vector_add(x, y))
        print(f"triton: {triton_ms:.4f} ms, {gbps(bytes_moved, triton_ms):.1f} GB/s")


if __name__ == "__main__":
    main()

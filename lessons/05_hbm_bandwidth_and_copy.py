from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.hardware import human_bytes
from triton_flash_course.kernels.memory_ops import copy_1d
from triton_flash_course.utils import benchmark_ms, gbps, parse_dtype, require_gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 26)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()

    require_gpu()
    dtype = parse_dtype(args.dtype)
    x = torch.randn(args.numel, device="cuda", dtype=dtype)

    y_ref = x.clone()
    y_tri = copy_1d(x, block_size=args.block_size)
    torch.testing.assert_close(y_tri, y_ref)

    bytes_moved = 2 * x.numel() * x.element_size()
    ms_torch = benchmark_ms(lambda: x.clone())
    ms_triton = benchmark_ms(lambda: copy_1d(x, block_size=args.block_size))

    print(f"Tensor logical size: {human_bytes(x.numel() * x.element_size())}")
    print(f"Approx bytes moved per copy: {human_bytes(bytes_moved)} = read + write")
    print(f"torch.clone : {ms_torch:.4f} ms, {gbps(bytes_moved, ms_torch):.1f} GB/s")
    print(f"triton copy : {ms_triton:.4f} ms, {gbps(bytes_moved, ms_triton):.1f} GB/s")


if __name__ == "__main__":
    main()

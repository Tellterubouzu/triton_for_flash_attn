from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.blockptr_ops import copy_2d_block_ptr
from triton_flash_course.utils import benchmark_ms, gbps, parse_dtype, require_gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1025)
    parser.add_argument("--n", type=int, default=2049)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    args = parser.parse_args()

    require_gpu()
    dtype = parse_dtype(args.dtype)
    x = torch.randn(args.m, args.n, device="cuda", dtype=dtype)
    y = copy_2d_block_ptr(x, block_m=args.block_m, block_n=args.block_n)
    torch.testing.assert_close(y, x)

    bytes_moved = 2 * x.numel() * x.element_size()
    ms = benchmark_ms(lambda: copy_2d_block_ptr(x, block_m=args.block_m, block_n=args.block_n), warmup=20, rep=80)
    print(f"copy_2d_block_ptr: {ms:.4f} ms, {gbps(bytes_moved, ms):.1f} GB/s")
    print("\nmake_block_ptr packs base, shape, strides, offsets, block_shape, and order into a block descriptor.")
    print("boundary_check=(0,1) handles edge tiles without constructing an explicit boolean mask tensor.")
    print("FlashAttention kernels often use block pointers for Q/K/V/O tiles and tl.advance to move K/V blocks.")


if __name__ == "__main__":
    main()

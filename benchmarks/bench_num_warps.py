from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch
from tabulate import tabulate

from triton_flash_course.kernels.matmul import matmul
from triton_flash_course.reference import matmul_ref
from triton_flash_course.utils import assert_close, benchmark_ms, is_gpu_ready, parse_dtype, tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=32)
    args = parser.parse_args()

    if not is_gpu_ready():
        raise RuntimeError("This benchmark requires a CUDA/ROCm GPU and Triton.")
    dtype = parse_dtype(args.dtype)
    a = torch.randn(args.m, args.k, device="cuda", dtype=dtype)
    b = torch.randn(args.k, args.n, device="cuda", dtype=dtype)
    ref = matmul_ref(a, b).float()
    flops = 2 * args.m * args.n * args.k

    rows = []
    for nw in args.warps:
        out = matmul(
            a,
            b,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=nw,
        )
        assert_close(f"num_warps={nw}", out, ref, dtype=dtype)
        ms = benchmark_ms(
            lambda nw=nw: matmul(
                a,
                b,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                num_warps=nw,
            ),
            rep=50,
        )
        rows.append([nw, f"{ms:.4f}", f"{tflops(flops, ms):.2f}"])

    print(tabulate(rows, headers=["num_warps", "ms", "TFLOP/s"]))
    print("注意: この matmul は教育用です。PyTorch/cuBLAS/rocBLAS に勝つことではなく、warps sweep の読み方が目的です。")


if __name__ == "__main__":
    main()

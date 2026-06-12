from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

import torch

from triton_flash_course.kernels.matmul import matmul
from triton_flash_course.reference import matmul_ref
from triton_flash_course.utils import benchmark_ms, get_device, is_gpu_ready, parse_dtype, tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="fp16")
    args = parser.parse_args()
    device = get_device()
    dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        dtype = torch.float32
    a = torch.randn(args.m, args.k, device=device, dtype=dtype)
    b = torch.randn(args.k, args.n, device=device, dtype=dtype)
    flops = 2 * args.m * args.n * args.k
    torch_ms = benchmark_ms(lambda: matmul_ref(a, b), warmup=5, rep=20)
    print(f"torch: {torch_ms:.4f} ms, {tflops(flops, torch_ms):.2f} TFLOP/s")
    if is_gpu_ready() and dtype != torch.float32:
        triton_ms = benchmark_ms(lambda: matmul(a, b), warmup=5, rep=20)
        print(f"triton: {triton_ms:.4f} ms, {tflops(flops, triton_ms):.2f} TFLOP/s")


if __name__ == "__main__":
    main()

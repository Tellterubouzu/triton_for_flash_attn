from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

import torch

from triton_flash_course.kernels.layernorm import layernorm
from triton_flash_course.reference import layernorm_ref
from triton_flash_course.utils import benchmark_ms, get_device, is_gpu_ready, parse_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="fp32")
    args = parser.parse_args()
    device = get_device()
    dtype = parse_dtype(args.dtype)
    x = torch.randn(args.rows, args.cols, device=device, dtype=dtype)
    w = torch.randn(args.cols, device=device, dtype=dtype)
    b = torch.randn(args.cols, device=device, dtype=dtype)
    torch_ms = benchmark_ms(lambda: layernorm_ref(x, w, b), warmup=5, rep=20)
    print(f"torch: {torch_ms:.4f} ms")
    if is_gpu_ready():
        triton_ms = benchmark_ms(lambda: layernorm(x, w, b), warmup=5, rep=20)
        print(f"triton: {triton_ms:.4f} ms")


if __name__ == "__main__":
    main()

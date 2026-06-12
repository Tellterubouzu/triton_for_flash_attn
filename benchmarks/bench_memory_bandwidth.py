from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch
from tabulate import tabulate

from triton_flash_course.kernels.memory_ops import copy_1d, zero_1d_like
from triton_flash_course.utils import BenchmarkResult, benchmark_ms, gbps, parse_dtype, require_gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 27)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()
    require_gpu()
    dtype = parse_dtype(args.dtype)
    x = torch.randn(args.numel, device="cuda", dtype=dtype)
    copy_bytes = 2 * x.numel() * x.element_size()
    zero_bytes = x.numel() * x.element_size()

    rows = []
    ms = benchmark_ms(lambda: x.clone(), warmup=20, rep=100)
    rows.append(BenchmarkResult("torch.clone", ms, gbps(copy_bytes, ms)).as_row())
    ms = benchmark_ms(lambda: copy_1d(x), warmup=20, rep=100)
    rows.append(BenchmarkResult("triton.copy_1d", ms, gbps(copy_bytes, ms)).as_row())
    ms = benchmark_ms(lambda: torch.zeros_like(x), warmup=20, rep=100)
    rows.append(BenchmarkResult("torch.zeros_like", ms, gbps(zero_bytes, ms)).as_row())
    ms = benchmark_ms(lambda: zero_1d_like(x), warmup=20, rep=100)
    rows.append(BenchmarkResult("triton.zero", ms, gbps(zero_bytes, ms)).as_row())

    print(tabulate(rows, headers=["name", "ms", "GB/s", "TFLOP/s"]))


if __name__ == "__main__":
    main()

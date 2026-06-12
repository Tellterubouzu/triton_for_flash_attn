from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch
from tabulate import tabulate

from triton_flash_course.kernels.memory_ops import strided_read
from triton_flash_course.utils import benchmark_ms, gbps, parse_dtype, require_gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 27)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()
    require_gpu()
    dtype = parse_dtype(args.dtype)
    x = torch.randn(args.numel, device="cuda", dtype=dtype)
    rows = []
    for stride in [1, 2, 4, 8, 16, 32, 64, 128]:
        y = strided_read(x, stride=stride)
        bytes_useful = y.numel() * x.element_size() + y.numel() * y.element_size()
        ms = benchmark_ms(lambda s=stride: strided_read(x, stride=s), warmup=20, rep=80)
        rows.append([stride, y.numel(), f"{ms:.4f}", f"{gbps(bytes_useful, ms):.1f}"])
    print(tabulate(rows, headers=["stride", "outputs", "ms", "useful GB/s"]))


if __name__ == "__main__":
    main()

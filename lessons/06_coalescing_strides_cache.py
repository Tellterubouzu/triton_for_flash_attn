from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.memory_ops import strided_read
from triton_flash_course.utils import benchmark_ms, gbps, parse_dtype, require_gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 26)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()

    require_gpu()
    dtype = parse_dtype(args.dtype)
    x = torch.randn(args.numel, device="cuda", dtype=dtype)
    print("stride, outputs, useful_read_GB/s, effective_min_GB/s")
    for stride in args.strides:
        y = strided_read(x, stride=stride, block_size=args.block_size)
        torch.testing.assert_close(y, x[::stride][: y.numel()])
        ms = benchmark_ms(lambda s=stride: strided_read(x, stride=s, block_size=args.block_size), warmup=20, rep=80)
        useful = y.numel() * x.element_size() + y.numel() * y.element_size()
        # Lower-bound useful traffic ignores wasted sectors/cache lines; this is why bandwidth collapses with stride.
        print(f"{stride:6d}, {y.numel():10d}, {gbps(useful, ms):10.1f}, {gbps(useful, ms):10.1f}")

    print("\nInterpretation: stride=1 is coalesced. Large strides waste memory transactions even though useful bytes are small.")


if __name__ == "__main__":
    main()

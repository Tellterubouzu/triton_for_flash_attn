from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.utils import benchmark_ms, gbps, parse_dtype, require_gpu


def bench_event(fn, rep: int = 50) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 26)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    require_gpu()
    dtype = parse_dtype(args.dtype)
    x_pageable = torch.randn(args.numel, dtype=dtype)
    x_pinned = torch.empty(args.numel, dtype=dtype, pin_memory=True).normal_()
    y_gpu = torch.empty(args.numel, dtype=dtype, device="cuda")
    bytes_moved = args.numel * torch.empty((), dtype=dtype).element_size()

    ms_h2d_pageable = bench_event(lambda: y_gpu.copy_(x_pageable, non_blocking=False))
    ms_h2d_pinned_sync = bench_event(lambda: y_gpu.copy_(x_pinned, non_blocking=False))
    ms_h2d_pinned_async = bench_event(lambda: y_gpu.copy_(x_pinned, non_blocking=True))

    z_cpu_pinned = torch.empty(args.numel, dtype=dtype, pin_memory=True)
    ms_d2h_pinned_async = bench_event(lambda: z_cpu_pinned.copy_(y_gpu, non_blocking=True))

    print(f"H2D pageable sync : {ms_h2d_pageable:.4f} ms, {gbps(bytes_moved, ms_h2d_pageable):.1f} GB/s")
    print(f"H2D pinned sync   : {ms_h2d_pinned_sync:.4f} ms, {gbps(bytes_moved, ms_h2d_pinned_sync):.1f} GB/s")
    print(f"H2D pinned async  : {ms_h2d_pinned_async:.4f} ms, {gbps(bytes_moved, ms_h2d_pinned_async):.1f} GB/s")
    print(f"D2H pinned async  : {ms_d2h_pinned_async:.4f} ms, {gbps(bytes_moved, ms_d2h_pinned_async):.1f} GB/s")

    print("\nPinned + non_blocking enables overlap patterns, but measuring overlap requires separate streams and a workload to overlap with transfer.")


if __name__ == "__main__":
    main()

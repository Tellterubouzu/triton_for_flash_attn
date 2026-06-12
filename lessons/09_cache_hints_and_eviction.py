from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.memory_ops import copy_1d_with_cache_hint
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
    bytes_moved = 2 * x.numel() * x.element_size()

    configs = [
        ("default", "", "", ""),
        ("load.ca", ".ca", "", ""),
        ("load.cg", ".cg", "", ""),
        ("load.cv", ".cv", "", ""),
        ("stream", ".cg", ".cs", "evict_first"),
        ("keep", ".ca", ".wb", "evict_last"),
    ]
    print("name, load_cache, store_cache, eviction, ms, GB/s")
    ref = x.clone()
    for name, load_cache, store_cache, eviction in configs:
        try:
            y = copy_1d_with_cache_hint(
                x,
                block_size=args.block_size,
                load_cache=load_cache,
                store_cache=store_cache,
                eviction=eviction,
            )
            torch.testing.assert_close(y, ref)
            ms = benchmark_ms(
                lambda lc=load_cache, sc=store_cache, ev=eviction: copy_1d_with_cache_hint(
                    x, block_size=args.block_size, load_cache=lc, store_cache=sc, eviction=ev
                ),
                warmup=20,
                rep=80,
            )
            print(f"{name}, {load_cache!r}, {store_cache!r}, {eviction!r}, {ms:.4f}, {gbps(bytes_moved, ms):.1f}")
        except Exception as exc:
            print(f"{name}, unsupported or failed: {exc}")

    print("\nCache hints are not portable performance guarantees. Inspect generated IR/PTX and profile counters.")


if __name__ == "__main__":
    main()

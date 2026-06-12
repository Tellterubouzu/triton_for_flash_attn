from __future__ import annotations

import argparse
import gc

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.hardware import allocator_snapshot, human_bytes
from triton_flash_course.utils import parse_dtype, require_gpu


def show(label: str) -> None:
    snap = allocator_snapshot()
    print(label)
    for k, v in snap.items():
        print(f"  {k}: {human_bytes(v) if isinstance(v, int) else v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 26)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()
    require_gpu()
    dtype = parse_dtype(args.dtype)
    torch.cuda.reset_peak_memory_stats()
    show("initial")
    xs = [torch.empty(args.numel, device="cuda", dtype=dtype) for _ in range(4)]
    show("after four allocations")
    del xs[:2]
    gc.collect()
    torch.cuda.synchronize()
    show("after deleting two tensors")
    torch.cuda.empty_cache()
    show("after empty_cache")
    del xs
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    show("after deleting all + empty_cache")


if __name__ == "__main__":
    main()

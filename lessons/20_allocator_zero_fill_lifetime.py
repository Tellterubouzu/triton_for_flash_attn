from __future__ import annotations

import argparse
import gc

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.hardware import allocator_snapshot, human_bytes
from triton_flash_course.kernels.memory_ops import zero_1d_like
from triton_flash_course.utils import benchmark_ms, gbps, parse_dtype, require_gpu


def show(label: str) -> None:
    snap = allocator_snapshot()
    if not snap.get("cuda_available"):
        print(label, snap)
        return
    print(
        f"{label}: allocated={human_bytes(snap['memory_allocated'])}, "
        f"reserved={human_bytes(snap['memory_reserved'])}, "
        f"max_allocated={human_bytes(snap['max_memory_allocated'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 26)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    require_gpu()
    dtype = parse_dtype(args.dtype)
    torch.cuda.reset_peak_memory_stats()
    show("initial")

    x = torch.empty(args.numel, device="cuda", dtype=dtype)
    show("after torch.empty")
    bytes_write = x.numel() * x.element_size()

    ms_zero_ = benchmark_ms(lambda: x.zero_(), warmup=20, rep=80)
    ms_zeros_like = benchmark_ms(lambda: torch.zeros_like(x), warmup=20, rep=80)
    ms_triton_zero = benchmark_ms(lambda: zero_1d_like(x), warmup=20, rep=80)

    print(f"zero_          : {ms_zero_:.4f} ms, min write BW {gbps(bytes_write, ms_zero_):.1f} GB/s")
    print(f"torch.zeros_like: {ms_zeros_like:.4f} ms, min write BW {gbps(bytes_write, ms_zeros_like):.1f} GB/s")
    print(f"triton zero    : {ms_triton_zero:.4f} ms, min write BW {gbps(bytes_write, ms_triton_zero):.1f} GB/s")

    del x
    gc.collect()
    torch.cuda.synchronize()
    show("after del + gc")
    torch.cuda.empty_cache()
    show("after empty_cache")
    print("\nInterpretation: del frees Tensor references; the PyTorch caching allocator may keep reserved blocks for reuse.")


if __name__ == "__main__":
    main()

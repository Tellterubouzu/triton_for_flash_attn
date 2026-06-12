from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

import torch

from triton_flash_course.profiling import profile_callable
from triton_flash_course.reference import attention_ref, attention_torch_sdpa
from triton_flash_course.utils import get_device, make_qkv, parse_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    device = get_device()
    dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        dtype = torch.float32
    q, k, v = make_qkv(args.batch, args.heads, args.seq, args.dim, dtype=dtype, device=device)
    print("=== naive attention ===")
    print(profile_callable(lambda: attention_ref(q, k, v, causal=args.causal), name="naive_attention", trace_dir=args.trace_dir))
    print("=== torch SDPA ===")
    print(profile_callable(lambda: attention_torch_sdpa(q, k, v, causal=args.causal), name="sdpa", trace_dir=args.trace_dir))
    if args.trace_dir:
        print(f"trace dir: {args.trace_dir}")


if __name__ == "__main__":
    main()

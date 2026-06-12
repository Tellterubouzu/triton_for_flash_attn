"""Lesson 16: bottleneck lab and profiler.

目的は、Triton kernel を書く前に「何が遅いのか」を特定することです。
この lesson では、naive attention を PyTorch profiler で見て、QK^T、softmax、PV、中間行列の memory 使用を観察します。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse
import math

import torch

from triton_flash_course.profiling import profile_callable
from triton_flash_course.reference import attention_ref, estimate_attention_bytes
from triton_flash_course.utils import get_device, make_qkv, parse_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    args = parser.parse_args()

    device = get_device()
    dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        dtype = torch.float32
    q, k, v = make_qkv(args.batch, args.heads, args.seq, args.dim, dtype=dtype, device=device)

    bytes_est = estimate_attention_bytes(args.batch, args.heads, args.seq, args.dim, dtype)
    print("memory lower-bound estimate:")
    for name, value in bytes_est.items():
        print(f"  {name:28s}: {value / 1024**2:.1f} MiB")

    sm_scale = 1.0 / math.sqrt(args.dim)
    table = profile_callable(lambda: attention_ref(q, k, v, sm_scale=sm_scale), name="naive_attention")
    print(table)
    print("判断基準: 巨大な [B,H,N,N] scores/probs が支配的なら、FlashAttention 型の融合対象です。")


if __name__ == "__main__":
    main()

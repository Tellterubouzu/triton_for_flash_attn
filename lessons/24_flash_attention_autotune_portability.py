"""Lesson 24: FlashAttention autotune and portability.

目的は `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` が GPU と shape に依存して効くことを実測することです。
Triton の `autotune` decorator だけでなく、まずは明示的 sweep を書くと性能モデルを理解しやすくなります。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

import torch

from triton_flash_course.kernels.flash_attention import FlashConfig, flash_attention_forward
from triton_flash_course.reference import attention_ref, estimate_attention_flops
from triton_flash_course.utils import assert_close, benchmark_ms, get_device, is_gpu_ready, make_qkv, parse_dtype, tflops


CONFIGS = [
    FlashConfig(block_m=16, block_n=32, num_warps=4, num_stages=3),
    FlashConfig(block_m=16, block_n=64, num_warps=4, num_stages=3),
    FlashConfig(block_m=32, block_n=32, num_warps=4, num_stages=3),
    FlashConfig(block_m=32, block_n=64, num_warps=4, num_stages=3),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    if not is_gpu_ready():
        print("この lesson は GPU/Triton が必要です。")
        return

    device = get_device()
    dtype = parse_dtype(args.dtype)
    if dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("FlashAttention autotune lesson は fp16/bf16 を対象にします。")
    q, k, v = make_qkv(args.batch, args.heads, args.seq, args.dim, dtype=dtype, device=device)
    ref = attention_ref(q, k, v, causal=args.causal)
    flops = estimate_attention_flops(args.batch, args.heads, args.seq, args.dim)

    rows: list[tuple[FlashConfig, float, float]] = []
    for cfg in CONFIGS:
        out = flash_attention_forward(q, k, v, causal=args.causal, config=cfg)
        assert_close(f"cfg={cfg}", out, ref, dtype=dtype)
        ms = benchmark_ms(lambda cfg=cfg: flash_attention_forward(q, k, v, causal=args.causal, config=cfg), warmup=5, rep=20)
        rows.append((cfg, ms, tflops(flops, ms)))

    rows.sort(key=lambda x: x[1])
    print("config sweep results:")
    for cfg, ms, tf in rows:
        print(f"  {cfg}: {ms:.4f} ms, {tf:.2f} TFLOP/s")
    print("次の課題: GPU 名、dtype、seq、dim ごとに最良 config が変わる理由を docs/reference/05_multi_gpu_portability.md に沿って整理してください。")


if __name__ == "__main__":
    main()

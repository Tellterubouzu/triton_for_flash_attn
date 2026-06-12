"""Lesson 23: FlashAttention forward in Triton.

この数式は block-wise / online softmax の更新を表します。

    m_new = max(m_old, max(scores_block))
    l_new = exp(m_old - m_new) * l_old + sum(exp(scores_block - m_new))
    acc_new = exp(m_old - m_new) * acc_old + exp(scores_block - m_new) V_block
    O = acc / l

各 Query block について Key/Value block を順に読み、softmax probability を HBM に保存せずに出力を作ります。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

import torch

from triton_flash_course.kernels.flash_attention import FlashConfig, flash_attention_forward
from triton_flash_course.reference import attention_ref, estimate_attention_flops
from triton_flash_course.utils import assert_close, benchmark_ms, get_device, is_gpu_ready, make_qkv, parse_dtype, tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=64)
    args = parser.parse_args()

    if not is_gpu_ready():
        print("この lesson は GPU/Triton が必要です。CPU では lessons/21_attention_pytorch_baselines.py を確認してください。")
        return

    device = get_device()
    dtype = parse_dtype(args.dtype)
    if dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("FlashAttention lesson は fp16/bf16 を対象にします。")
    q, k, v = make_qkv(args.batch, args.heads, args.seq, args.dim, dtype=dtype, device=device)
    ref = attention_ref(q, k, v, causal=args.causal)
    cfg = FlashConfig(block_m=args.block_m, block_n=args.block_n)
    out = flash_attention_forward(q, k, v, causal=args.causal, config=cfg)
    assert_close("flash_attention_forward", out, ref, dtype=dtype)

    flops = estimate_attention_flops(args.batch, args.heads, args.seq, args.dim)
    naive_ms = benchmark_ms(lambda: attention_ref(q, k, v, causal=args.causal), warmup=5, rep=20)
    flash_ms = benchmark_ms(lambda: flash_attention_forward(q, k, v, causal=args.causal, config=cfg), warmup=5, rep=20)
    print(f"naive PyTorch: {naive_ms:.4f} ms, {tflops(flops, naive_ms):.2f} TFLOP/s")
    print(f"triton flash : {flash_ms:.4f} ms, {tflops(flops, flash_ms):.2f} TFLOP/s")
    print("観察: seq が大きいほど中間行列を HBM に書かない利点が出ます。ただし教育用 kernel は production 実装ではありません。")


if __name__ == "__main__":
    main()

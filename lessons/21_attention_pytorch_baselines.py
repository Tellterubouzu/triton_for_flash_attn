"""Lesson 21: PyTorch attention baselines.

この数式は scaled dot-product attention を表します。

    O = softmax(QK^T / sqrt(d)) V

ここでは naive 実装と `torch.nn.functional.scaled_dot_product_attention` を比較します。
naive 実装は中間行列を materialize するため、FlashAttention の必要性を観察しやすい baseline です。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

import torch

from triton_flash_course.reference import attention_ref, attention_torch_sdpa, estimate_attention_flops
from triton_flash_course.utils import assert_close, benchmark_ms, get_device, make_qkv, parse_dtype, tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    device = get_device()
    dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        dtype = torch.float32
    q, k, v = make_qkv(args.batch, args.heads, args.seq, args.dim, dtype=dtype, device=device)
    ref = attention_ref(q, k, v, causal=args.causal)
    sdpa = attention_torch_sdpa(q, k, v, causal=args.causal)
    assert_close("sdpa", sdpa, ref, dtype=dtype)

    flops = estimate_attention_flops(args.batch, args.heads, args.seq, args.dim)
    naive_ms = benchmark_ms(lambda: attention_ref(q, k, v, causal=args.causal), warmup=5, rep=20)
    sdpa_ms = benchmark_ms(lambda: attention_torch_sdpa(q, k, v, causal=args.causal), warmup=5, rep=20)
    print(f"naive PyTorch: {naive_ms:.4f} ms, {tflops(flops, naive_ms):.2f} TFLOP/s")
    print(f"torch SDPA   : {sdpa_ms:.4f} ms, {tflops(flops, sdpa_ms):.2f} TFLOP/s")
    print("観察: PyTorch SDPA は backend 選択により既に fused/flash 系になる場合があります。教材では naive baseline との差を見ます。")


if __name__ == "__main__":
    main()

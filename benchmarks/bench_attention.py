from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

import torch

from triton_flash_course.kernels.flash_attention import FlashConfig, flash_attention_forward
from triton_flash_course.reference import attention_ref, attention_torch_sdpa, estimate_attention_flops
from triton_flash_course.utils import benchmark_ms, get_device, is_gpu_ready, make_qkv, parse_dtype, tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        dtype = torch.float32
    q, k, v = make_qkv(args.batch, args.heads, args.seq, args.dim, dtype=dtype, device=device)
    flops = estimate_attention_flops(args.batch, args.heads, args.seq, args.dim)

    naive_ms = benchmark_ms(lambda: attention_ref(q, k, v, causal=args.causal), warmup=5, rep=20)
    sdpa_ms = benchmark_ms(lambda: attention_torch_sdpa(q, k, v, causal=args.causal), warmup=5, rep=20)
    print(f"naive PyTorch: {naive_ms:.4f} ms, {tflops(flops, naive_ms):.2f} TFLOP/s")
    print(f"torch SDPA   : {sdpa_ms:.4f} ms, {tflops(flops, sdpa_ms):.2f} TFLOP/s")

    if is_gpu_ready() and dtype in {torch.float16, torch.bfloat16}:
        cfg = FlashConfig(block_m=args.block_m, block_n=args.block_n)
        flash_ms = benchmark_ms(lambda: flash_attention_forward(q, k, v, causal=args.causal, config=cfg), warmup=5, rep=20)
        print(f"triton flash : {flash_ms:.4f} ms, {tflops(flops, flash_ms):.2f} TFLOP/s, cfg={cfg}")


if __name__ == "__main__":
    main()

"""Lesson 12: numerics and correctness policy.

この lesson は、Triton kernel の結果を PyTorch reference と比較するときの
許容誤差、softmax の数値安定化、dtype ごとの誤差の見方を扱います。

FlashAttention では softmax を明示 materialize しないため、演算順序が naive
PyTorch と一致しません。bitwise 一致ではなく、dtype と演算内容に応じた
correctness policy を先に決めることが重要です。
"""
from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.numerics import error_report, recommended_tolerance, softmax_stability_summary
from triton_flash_course.reference import attention_ref
from triton_flash_course.utils import get_device, make_qkv, parse_dtype


def _print_report(title: str, report: object) -> None:
    print(f"\n{title}")
    if hasattr(report, "to_dict"):
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    device = get_device()
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        print("CPU では fp16/bf16 attention の演算 coverage が環境依存なので fp32 に切り替えます。")
        dtype = torch.float32

    print("Recommended tolerances")
    for op in ["generic", "softmax", "attention", "layernorm"]:
        tol = recommended_tolerance(dtype, op=op)
        print(f"  {op:10s}: rtol={tol.rtol:g}, atol={tol.atol:g}  # {tol.note}")

    x = torch.tensor([[1000.0, 999.0, 998.0], [1.0, 2.0, 3.0]], device=device)
    _print_report("Softmax stability summary", softmax_stability_summary(x))

    q, k, v = make_qkv(args.batch, args.heads, args.seq, args.dim, dtype=dtype, device=device, seed=0)
    out_dtype = attention_ref(q, k, v, causal=args.causal)
    out_fp32 = attention_ref(q.float(), k.float(), v.float(), causal=args.causal).to(out_dtype.dtype)
    report = error_report("attention_dtype_vs_fp32", out_dtype, out_fp32, dtype=dtype, op="attention")
    _print_report("Attention error report", report)

    print("\nInterpretation")
    print("- stable softmax は logits から row max を引くため overflow を避けます。")
    print("- fp16/bf16 kernel は fp32 accumulator を使っても、演算順序差で PyTorch naive と完全一致しません。")
    print("- FlashAttention の test では shape, dtype, causal, head_dim ごとに tolerance を明示します。")


if __name__ == "__main__":
    main()

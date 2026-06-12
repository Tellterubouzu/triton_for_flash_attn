"""Lesson 25: FlashAttention validation matrix.

単一 shape の correctness だけでは、FlashAttention kernel の実装品質は判断できません。
この lesson では dtype, sequence length, head_dim, causal flag を sweep し、どの条件で
pass/fail/skip したかを表にします。
"""
from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.numerics import error_report
from triton_flash_course.portability import validation_matrix
from triton_flash_course.reference import attention_ref
from triton_flash_course.utils import get_device, is_gpu_ready, make_qkv, parse_dtype


def _try_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    from triton_flash_course.kernels.flash_attention import flash_attention_forward

    return flash_attention_forward(q, k, v, causal=causal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--dtype", action="append", choices=["fp16", "bf16", "fp32"], default=None)
    args = parser.parse_args()

    device = get_device()
    dtypes = tuple(args.dtype) if args.dtype else ("fp16", "bf16")
    cases = validation_matrix(dtypes=dtypes, include_large=args.include_large)[: args.max_cases]
    gpu_ready = is_gpu_ready()

    rows = []
    for case in cases:
        dtype = parse_dtype(case.dtype)
        if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
            rows.append({**case.to_dict(), "status": "skip", "reason": "CPU fallback; GPU/Triton required"})
            continue
        q, k, v = make_qkv(case.batch, case.heads, case.seq, case.dim, dtype=dtype, device=device, seed=case.seq + case.dim)
        ref = attention_ref(q, k, v, causal=case.causal)
        if not gpu_ready:
            rows.append({**case.to_dict(), "status": "reference-only", "reason": "Triton/GPU unavailable"})
            continue
        try:
            actual = _try_flash(q, k, v, case.causal)
            report = error_report("flash_attention", actual, ref, dtype=dtype, op="attention")
            rows.append({**case.to_dict(), "status": "pass" if report.passed else "fail", **report.to_dict()})
        except Exception as exc:  # noqa: BLE001 - education script should record failures.
            rows.append({**case.to_dict(), "status": "error", "reason": type(exc).__name__, "message": str(exc)[:240]})

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\nNext step: failing rows should become pytest parametrized cases before tuning performance.")


if __name__ == "__main__":
    main()

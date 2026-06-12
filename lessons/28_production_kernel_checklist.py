"""Lesson 28: production FlashAttention kernel checklist.

教育用 forward kernel から production kernel へ進むときに必要な作業を、
correctness, performance, portability, integration の観点で棚卸しします。
"""
from __future__ import annotations

import json

from _bootstrap import add_src_to_path

add_src_to_path()


CHECKLIST = {
    "correctness": [
        "dtype matrix: fp16, bf16, fp32/tf32 policy, optionally fp8",
        "shape matrix: seq=1, non-power-of-two seq, head_dim=16/32/64/128, odd batch/head counts",
        "causal and non-causal masks, empty/short sequence behavior",
        "NaN/Inf policy and deterministic mode policy",
        "forward/backward gradient checks against PyTorch reference",
    ],
    "performance": [
        "separate warmup and measurement; synchronize correctly",
        "sweep BLOCK_M, BLOCK_N, num_warps, num_stages per GPU family",
        "verify tensor core/MFMA lowering for QK^T and PV",
        "inspect HBM bandwidth, L2 hit rate, register spills, achieved occupancy",
        "record launch overhead and small-sequence regime separately",
    ],
    "features": [
        "dropout forward/backward with reproducible RNG state",
        "variable length / packed sequence support",
        "GQA/MQA head mapping",
        "KV cache / paged attention for decoding",
        "bias, ALiBi, sliding window, local attention variants if needed",
    ],
    "portability": [
        "NVIDIA and AMD CI jobs, or at least saved benchmark matrix per GPU",
        "backend-specific config defaults without backend-specific intrinsics in core logic",
        "ROCm wavefront and MFMA behavior checked separately from CUDA warp/MMA behavior",
        "version pins for torch, triton, driver, CUDA/ROCm, benchmark scripts",
    ],
    "integration": [
        "torch.compile behavior and graph breaks",
        "torch.library or custom autograd wrapper if training is required",
        "fallback path to torch.nn.functional.scaled_dot_product_attention",
        "clear error messages for unsupported dtype/shape/device",
    ],
}


def main() -> None:
    print(json.dumps(CHECKLIST, indent=2, ensure_ascii=False))
    print("\nRule: correctness matrix before autotune; profiler counters before micro-optimizing code style.")


if __name__ == "__main__":
    main()

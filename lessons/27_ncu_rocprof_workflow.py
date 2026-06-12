"""Lesson 27: external profiler workflow for NVIDIA Nsight Compute and AMD rocprof.

PyTorch profiler は operator と CUDA/HIP kernel の入口を見つける道具です。
最終的な kernel 改善では、Nsight Compute や rocprof で memory throughput,
occupancy, stall reason, Tensor Core/MFMA utilization を確認します。
"""
from __future__ import annotations

import argparse
import json
import shutil

from _bootstrap import add_src_to_path

add_src_to_path()

from triton_flash_course.portability import backend_from_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="benchmarks/bench_attention.py --seq 2048 --dim 64 --dtype fp16")
    parser.add_argument("--kernel-name", default="flash")
    args = parser.parse_args()

    backend = backend_from_torch()
    commands = {
        "nvidia_nsight_compute_summary": (
            "ncu --set full --target-processes all "
            f"--kernel-name-base demangled --kernel-name regex:{args.kernel_name} "
            f"python {args.target}"
        ),
        "nvidia_nsight_compute_sections": (
            "ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats "
            f"--target-processes all python {args.target}"
        ),
        "nvidia_nsight_systems_timeline": f"nsys profile --trace=cuda,nvtx,osrt -o traces/nsys_flash python {args.target}",
        "amd_rocprof_compute": f"rocprof-compute profile -- python {args.target}",
        "amd_rocprof_timeline": f"rocprof --hip-trace --hsa-trace -o traces/rocprof.csv python {args.target}",
    }
    availability = {name: shutil.which(name) is not None for name in ["ncu", "nsys", "rocprof", "rocprof-compute"]}
    print(json.dumps({"detected_backend": backend, "tool_availability": availability, "commands": commands}, indent=2, ensure_ascii=False))

    print("\nMetrics to inspect")
    print("- Memory: achieved HBM bandwidth, L2 hit rate, global load/store transactions, replay/sector waste.")
    print("- Compute: tensor pipe utilization, mma/wgmma/mfma instruction counts, fp32 pipe usage for exp/reduction.")
    print("- Scheduling: occupancy, eligible warps per cycle, long scoreboard stalls, barrier stalls, register spills.")
    print("- FlashAttention-specific: Q/K/V load balance, causal boundary blocks, BLOCK_M/BLOCK_N sweep, num_warps sweep.")


if __name__ == "__main__":
    main()

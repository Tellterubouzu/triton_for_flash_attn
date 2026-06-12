from __future__ import annotations

import runpy
import sys
from pathlib import Path

LESSONS: list[tuple[str, list[str]]] = [
    ("lessons/00_setup_and_sanity.py", []),
    ("lessons/01_gpu_execution_model.py", ["--numel", "35", "--block-size", "8"]),
    ("lessons/02_warps_sms_occupancy.py", ["--warps", "1", "2", "4"]),
    ("lessons/03_memory_hierarchy_and_device_query.py", []),
    ("lessons/04_tensor_addresses_strides_layouts.py", []),
    ("lessons/07_vector_add_first_kernel.py", []),
    ("lessons/08_elementwise_fusion.py", []),
    ("lessons/11_reductions_and_softmax.py", []),
    ("lessons/12_numerics_and_correctness.py", ["--seq", "16", "--dim", "16", "--dtype", "fp32"]),
    ("lessons/15_layernorm_and_rowwise_norm.py", []),
    ("lessons/16_bottleneck_lab_and_profiler.py", ["--heads", "1", "--seq", "32", "--dim", "16", "--dtype", "fp32"]),
    ("lessons/21_attention_pytorch_baselines.py", ["--heads", "1", "--seq", "32", "--dim", "16", "--dtype", "fp32"]),
    ("lessons/22_flash_attention_io_accounting.py", ["--heads", "4", "--seq", "128", "--dim", "32"]),
    ("lessons/25_flash_attention_validation_matrix.py", ["--max-cases", "4", "--dtype", "fp32"]),
    ("lessons/26_flash_attention_backward_plan.py", []),
    ("lessons/27_ncu_rocprof_workflow.py", []),
    ("lessons/28_production_kernel_checklist.py", []),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    original_argv = sys.argv[:]
    try:
        for lesson, args in LESSONS:
            print(f"\n===== {lesson} {' '.join(args)} =====")
            lesson_dir = str((root / lesson).parent)
            if lesson_dir not in sys.path:
                sys.path.insert(0, lesson_dir)
            sys.argv = [lesson, *args]
            runpy.run_path(str(root / lesson), run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

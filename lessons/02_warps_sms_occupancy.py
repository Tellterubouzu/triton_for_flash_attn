"""Lesson 02: warps, SMs, and occupancy sketch.

この lesson は実 occupancy を正確に予測するものではありません。device properties から
`num_warps` を増やしたときに resident program の上限がどう下がり得るかを確認します。
"""
from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from tabulate import tabulate

from triton_flash_course.gpu_model import occupancy_sketch
from triton_flash_course.hardware import get_device_properties_dict, print_device_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    print("Device report")
    print_device_report()

    props = get_device_properties_dict()
    if not props:
        print(
            "\nGPU がないため、CUDA らしい仮想 device properties で概念デモを表示します。"
        )
        props = {
            "warp_size": 32,
            "max_threads_per_multi_processor": 2048,
            "max_warps_per_multiprocessor": 64,
            "max_blocks_per_multiprocessor": 32,
            "multi_processor_count": 108,
        }

    rows = []
    for nw in args.warps:
        sketch = occupancy_sketch(num_warps=nw, device_properties=props, backend="cuda")
        rows.append(
            [
                nw,
                sketch.requested_threads_per_program,
                sketch.resident_programs_by_threads,
                sketch.resident_programs_by_warps,
                sketch.resident_programs_by_blocks,
                sketch.resident_programs_upper_bound,
                "-"
                if sketch.theoretical_occupancy_fraction is None
                else f"{100.0 * sketch.theoretical_occupancy_fraction:.1f}%",
            ]
        )

    print("\nnum_warps sweep: upper-bound occupancy sketch")
    print(
        tabulate(
            rows,
            headers=[
                "num_warps",
                "threads/program",
                "by threads",
                "by warps",
                "by blocks",
                "resident upper",
                "occupancy upper",
            ],
        )
    )
    print(
        "\n注意: これは上限です。実際には register pressure, shared memory, num_stages, compiler schedule で下がります。"
    )
    print("\nraw example for first num_warps")
    print(json.dumps(occupancy_sketch(num_warps=args.warps[0], device_properties=props, backend="cuda").to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

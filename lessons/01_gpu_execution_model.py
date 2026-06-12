"""Lesson 01: kernel, launch, grid, Triton program, and warp.

この lesson は概念説明と小さな program_id trace から構成されます。
GPU/Triton がない環境では説明と launch geometry だけを表示します。
"""
from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.gpu_model import describe_1d_launch, pretty_print_mapping_table
from triton_flash_course.utils import is_gpu_ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=35)
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args()

    print("CUDA と Triton の実行単位対応")
    print(pretty_print_mapping_table())
    print()

    geometry = describe_1d_launch(args.numel, args.block_size)
    print("1D launch geometry")
    print(json.dumps(geometry.to_dict(), indent=2, ensure_ascii=False))
    print(
        "\nこの例では、1 Triton program instance が BLOCK_SIZE 個の logical element を担当します。"
    )

    if not is_gpu_ready():
        print("\nGPU/Triton がないため、program_id trace kernel は実行しません。")
        return

    from triton_flash_course.kernels.execution_model import program_id_trace

    pid, offsets = program_id_trace(args.numel, args.block_size)
    torch.cuda.synchronize()
    print("\nprogram_id written per element")
    print(pid.cpu().tolist())
    print("offset written per element")
    print(offsets.cpu().tolist())
    print(
        "\n観察: 最後の program は BLOCK_SIZE 個の offset を作りますが、mask により n_elements 以上の store は無効化されます。"
    )


if __name__ == "__main__":
    main()

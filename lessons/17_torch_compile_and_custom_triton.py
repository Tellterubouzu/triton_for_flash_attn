"""Lesson 17: torch.compile and custom Triton kernels.

目的は、Triton kernel を単体で呼ぶだけでなく、PyTorch 関数の一部として使う感覚を掴むことです。
`torch.compile` は graph break があると最適化機会を失います。custom kernel は「必要な箇所だけ」使います。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.vector_add import vector_add
from triton_flash_course.utils import assert_close, benchmark_ms, get_device, is_gpu_ready


def eager_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y) * 0.5


def triton_inside_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = vector_add(x, y)
    return torch.relu(z) * 0.5


def main() -> None:
    device = get_device()
    dtype = torch.float32
    x = torch.randn(2_000_000, device=device, dtype=dtype)
    y = torch.randn_like(x)
    ref = eager_fn(x, y)

    if not is_gpu_ready():
        print("GPU/Triton がないため PyTorch baseline だけ実行します。")
        print(ref[:5])
        return

    out = triton_inside_fn(x, y)
    assert_close("triton_inside_fn", out, ref, dtype=dtype)
    compiled_eager = torch.compile(eager_fn)
    eager_ms = benchmark_ms(lambda: eager_fn(x, y))
    compiled_ms = benchmark_ms(lambda: compiled_eager(x, y))
    mixed_ms = benchmark_ms(lambda: triton_inside_fn(x, y))
    print(f"eager          : {eager_ms:.4f} ms")
    print(f"torch.compile  : {compiled_ms:.4f} ms")
    print(f"manual triton  : {mixed_ms:.4f} ms")
    print("観察: 単純な graph は torch.compile が強いです。manual Triton は profiler で必要性を確認してから使います。")


if __name__ == "__main__":
    main()

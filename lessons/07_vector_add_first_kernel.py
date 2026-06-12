"""Lesson 07: first Triton kernel with vector add.

この数式は、同じ長さの 2 つのベクトルを要素ごとに足し合わせる操作を表します。

    c_i = a_i + b_i

FlashAttention に直接は出てきませんが、`program_id`, `tl.arange`, mask,
`tl.load`, `tl.store` は以降すべての kernel の最小単位です。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.vector_add import vector_add
from triton_flash_course.reference import vector_add_ref
from triton_flash_course.utils import assert_close, benchmark_ms, gbps, get_device, is_gpu_ready


def main() -> None:
    device = get_device()
    n = 8_000_000
    dtype = torch.float32
    a = torch.randn(n, device=device, dtype=dtype)
    b = torch.randn(n, device=device, dtype=dtype)
    ref = vector_add_ref(a, b)

    if not is_gpu_ready():
        print("GPU/Triton がないため PyTorch baseline だけ実行します。")
        print(ref[:5])
        return

    out = vector_add(a, b)
    assert_close("vector_add", out, ref, dtype=dtype)
    torch_ms = benchmark_ms(lambda: vector_add_ref(a, b))
    triton_ms = benchmark_ms(lambda: vector_add(a, b))
    bytes_moved = 3 * a.numel() * a.element_size()
    print(f"torch : {torch_ms:.4f} ms, {gbps(bytes_moved, torch_ms):.1f} GB/s")
    print(f"triton: {triton_ms:.4f} ms, {gbps(bytes_moved, triton_ms):.1f} GB/s")
    print("観察: この kernel はほぼ memory bandwidth bound です。")


if __name__ == "__main__":
    main()

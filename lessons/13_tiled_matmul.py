"""Lesson 13: tiled matrix multiplication.

この数式は、行列 A と B の積 C を表します。

    C_ij = sum_k A_ik B_kj

FlashAttention forward は QK^T と PV の 2 つの matmul を、softmax 統計量の更新と融合します。
ここではまず tile と `tl.dot` の基本だけを学びます。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.matmul import matmul
from triton_flash_course.reference import matmul_ref
from triton_flash_course.utils import assert_close, benchmark_ms, get_device, is_gpu_ready, tflops


def main() -> None:
    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    m, k, n = 1024, 1024, 1024
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    ref = matmul_ref(a, b).float()

    if not is_gpu_ready() or dtype == torch.float32:
        print("GPU/Triton がないため PyTorch baseline だけ実行します。")
        print(ref[:2, :2])
        return

    out = matmul(a, b)
    assert_close("matmul", out, ref, dtype=torch.float16)
    flops = 2 * m * n * k
    torch_ms = benchmark_ms(lambda: matmul_ref(a, b))
    triton_ms = benchmark_ms(lambda: matmul(a, b))
    print(f"torch : {torch_ms:.4f} ms, {tflops(flops, torch_ms):.2f} TFLOP/s")
    print(f"triton: {triton_ms:.4f} ms, {tflops(flops, triton_ms):.2f} TFLOP/s")
    print("観察: PyTorch matmul は極めて強い baseline です。自作 kernel は融合のために書く、という判断が重要です。")


if __name__ == "__main__":
    main()

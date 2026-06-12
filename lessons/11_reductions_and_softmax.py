"""Lesson 11: reductions and stable softmax.

この数式は、各行の logits を確率分布へ正規化する操作を表します。

    y_ij = exp(x_ij - m_i) / sum_k exp(x_ik - m_i)
    m_i = max_j x_ij

m_i を引くのは数値安定化です。FlashAttention の online softmax は、この row-wise
softmax を block 分割して逐次更新するものです。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.softmax import row_softmax
from triton_flash_course.reference import softmax_ref
from triton_flash_course.utils import assert_close, benchmark_ms, get_device, is_gpu_ready


def main() -> None:
    device = get_device()
    dtype = torch.float32
    x = torch.randn(4096, 1024, device=device, dtype=dtype)
    ref = softmax_ref(x)

    if not is_gpu_ready():
        print("GPU/Triton がないため PyTorch baseline だけ実行します。")
        print(ref[0, :5])
        return

    out = row_softmax(x)
    assert_close("row_softmax", out, ref, dtype=dtype)
    torch_ms = benchmark_ms(lambda: softmax_ref(x))
    triton_ms = benchmark_ms(lambda: row_softmax(x))
    print(f"torch : {torch_ms:.4f} ms")
    print(f"triton: {triton_ms:.4f} ms")
    print("観察: max / exp / sum / div を 1 pass 近くに融合できると HBM traffic が減ります。")


if __name__ == "__main__":
    main()

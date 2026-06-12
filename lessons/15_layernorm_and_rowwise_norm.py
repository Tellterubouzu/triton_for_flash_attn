"""Lesson 15: LayerNorm and row-wise normalization.

この数式は、各行の特徴次元を平均 0・分散 1 に正規化し、学習可能な weight/bias を適用する操作を表します。

    y = (x - E[x]) / sqrt(Var[x] + eps) * w + b

row-wise reduction と affine を 1 kernel に融合する練習です。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.layernorm import layernorm
from triton_flash_course.reference import layernorm_ref
from triton_flash_course.utils import assert_close, benchmark_ms, get_device, is_gpu_ready


def main() -> None:
    device = get_device()
    dtype = torch.float32
    rows, cols = 4096, 1024
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    w = torch.randn(cols, device=device, dtype=dtype)
    b = torch.randn(cols, device=device, dtype=dtype)
    ref = layernorm_ref(x, w, b)

    if not is_gpu_ready():
        print("GPU/Triton がないため PyTorch baseline だけ実行します。")
        print(ref[0, :5])
        return

    out = layernorm(x, w, b)
    assert_close("layernorm", out, ref, dtype=dtype)
    torch_ms = benchmark_ms(lambda: layernorm_ref(x, w, b))
    triton_ms = benchmark_ms(lambda: layernorm(x, w, b))
    print(f"torch : {torch_ms:.4f} ms")
    print(f"triton: {triton_ms:.4f} ms")
    print("観察: reduction の軸が 1 row に閉じている処理は Triton 化しやすいです。")


if __name__ == "__main__":
    main()

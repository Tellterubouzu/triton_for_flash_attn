"""Lesson 08: elementwise fusion.

この数式は、入力 x に sigmoid gate を掛ける Swish/SiLU 型の要素ごとの非線形変換を表します。

    y_i = x_i * sigmoid(beta * x_i)

PyTorch eager では `beta*x`, `sigmoid`, `x*...` が複数 operator として現れます。
Triton では load -> compute -> store を 1 kernel に融合します。
"""
from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.kernels.fused_ops import swish
from triton_flash_course.reference import swish_ref
from triton_flash_course.utils import assert_close, benchmark_ms, get_device, is_gpu_ready


def main() -> None:
    device = get_device()
    dtype = torch.float32
    x = torch.randn(16_000_000, device=device, dtype=dtype)
    ref = swish_ref(x, beta=1.0)

    if not is_gpu_ready():
        print("GPU/Triton がないため PyTorch baseline だけ実行します。")
        print(ref[:5])
        return

    out = swish(x)
    assert_close("swish", out, ref, dtype=dtype)

    compiled = torch.compile(swish_ref) if hasattr(torch, "compile") else swish_ref
    torch_ms = benchmark_ms(lambda: swish_ref(x))
    compile_ms = benchmark_ms(lambda: compiled(x))
    triton_ms = benchmark_ms(lambda: swish(x))
    print(f"torch eager : {torch_ms:.4f} ms")
    print(f"torch.compile: {compile_ms:.4f} ms")
    print(f"triton      : {triton_ms:.4f} ms")
    print("観察: 単純な elementwise は torch.compile が十分な場合もあります。")


if __name__ == "__main__":
    main()

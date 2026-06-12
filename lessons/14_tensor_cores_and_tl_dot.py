"""Lesson 14: Tensor Core / MFMA and tl.dot.

最小の `tl.dot` kernel を compile し、取得できる IR/PTX/ISA 文字列から
mma/wgmma/mfma/ldmatrix などの token を探します。最終判断は Nsight Compute などの
profiler metric で行ってください。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.gpu_model import summarize_compiled_asm
from triton_flash_course.kernels.execution_model import compile_tiny_dot
from triton_flash_course.reference import matmul_ref
from triton_flash_course.utils import assert_close, benchmark_ms, is_gpu_ready, parse_dtype, tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--input-precision", default="tf32", choices=["tf32", "tf32x3", "ieee"])
    parser.add_argument("--dump-dir", default="ir_dumps/tiny_dot")
    args = parser.parse_args()

    print("`tl.dot` は block-level matrix multiplication です。")
    print("NVIDIA では条件が合うと mma/wgmma 系、AMD では mfma 系へ lower され得ます。")

    if not is_gpu_ready():
        print("\nGPU/Triton がないため compile は実行しません。")
        print("GPU 環境では --dtype fp16 / bf16 / fp32 と --input-precision を変えて比較してください。")
        return

    dtype = parse_dtype(args.dtype)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU/PyTorch build does not report BF16 support.")

    out, compiled = compile_tiny_dot(
        M=args.m,
        N=args.n,
        K=args.k,
        dtype=dtype,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.num_warps,
        input_precision=args.input_precision,
    )
    torch.cuda.synchronize()

    # compile_tiny_dot creates private random inputs. For a transparent correctness
    # check we re-run a tiny deterministic matmul through the same kernel is not
    # exposed here; this lesson focuses on compiler output. We still print output
    # statistics to catch NaN/Inf.
    print("\noutput statistics")
    print({"mean": float(out.float().mean().item()), "std": float(out.float().std().item()), "has_nan": bool(torch.isnan(out).any().item())})

    print("\nAvailable asm keys:", list(compiled.asm.keys()))
    summary = summarize_compiled_asm(compiled.asm)
    print("\nInstruction-token summary")
    print(json.dumps({k: v.to_dict() for k, v in summary.items()}, indent=2, ensure_ascii=False))

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    for key, value in compiled.asm.items():
        suffix = "bin" if isinstance(value, bytes) else "txt"
        path = dump_dir / f"tiny_dot.{key}.{suffix}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(str(value), encoding="utf-8")
        print("wrote", path)

    # Separate benchmark with torch.matmul is included only as a calibration point.
    a = torch.randn(args.m, args.k, device="cuda", dtype=dtype)
    b = torch.randn(args.k, args.n, device="cuda", dtype=dtype)
    flops = 2 * args.m * args.n * args.k
    torch_ms = benchmark_ms(lambda: matmul_ref(a, b), rep=50)
    print(f"\nTorch matmul calibration: {torch_ms:.4f} ms, {tflops(flops, torch_ms):.2f} TFLOP/s")
    print("観察: generated code token と profiler metric を対応づけること。token count だけで性能判断しないでください。")


if __name__ == "__main__":
    main()

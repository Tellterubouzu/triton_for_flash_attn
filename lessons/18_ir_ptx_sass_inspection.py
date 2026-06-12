from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.utils import require_gpu

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _tiny_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca")
        tl.store(y_ptr + offsets, x + 1.0, mask=mask, cache_modifier=".wb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="ir_dumps")
    parser.add_argument("--numel", type=int, default=1024)
    args = parser.parse_args()

    require_gpu()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.randn(args.numel, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(args.numel, meta["BLOCK_SIZE"]),)
    compiled = _tiny_kernel[grid](x, y, args.numel, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    print("Available asm keys:", list(compiled.asm.keys()))
    for key, value in compiled.asm.items():
        if isinstance(value, bytes):
            path = out_dir / f"tiny_kernel.{key}.bin"
            path.write_bytes(value)
        else:
            path = out_dir / f"tiny_kernel.{key}.txt"
            path.write_text(str(value), encoding="utf-8")
        print("wrote", path)

    print("\nLook for load/store cache modifiers, vectorized memory ops, register count hints, and target-specific ISA.")


if __name__ == "__main__":
    main()

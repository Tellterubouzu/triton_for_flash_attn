from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.hardware import (
    address_from_storage_base,
    address_from_tensor_data_ptr,
    print_tensor_address_table,
    tensor_address_info,
)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.arange(4 * 5, device=device, dtype=torch.float32).reshape(4, 5)
    y = x[1:, ::2]

    print("Base contiguous tensor x")
    print_tensor_address_table(x, [(0, 0), (0, 1), (1, 0), (3, 4)])

    print("\nView y = x[1:, ::2]")
    print_tensor_address_table(y, [(0, 0), (0, 1), (1, 0), (2, 2)])

    print("\nCheck address formulas")
    for idx in [(0, 0), (0, 1), (1, 0), (2, 2)]:
        a = address_from_storage_base(y, idx)
        b = address_from_tensor_data_ptr(y, idx)
        print(f"  idx={idx}: storage_base_formula=0x{a:x}, data_ptr_formula=0x{b:x}, equal={a == b}")

    info = tensor_address_info(y)
    print("\nInterpretation:")
    print(f"  y.stride()={info.stride} means y[i,j] uses element offset i*{info.stride[0]} + j*{info.stride[1]} from y.data_ptr().")
    print("  In Triton, ptr + offset uses element offsets, not byte offsets.")

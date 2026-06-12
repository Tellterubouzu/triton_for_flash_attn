from __future__ import annotations

import torch

from triton_flash_course.hardware import (
    address_from_storage_base,
    address_from_tensor_data_ptr,
    linear_offset_elements,
    tensor_address_info,
)


def test_linear_offset_elements() -> None:
    assert linear_offset_elements((2, 3), (10, 2), 5) == 31


def test_address_formulas_match_for_view_cpu() -> None:
    x = torch.arange(6 * 7, dtype=torch.float32).reshape(6, 7)
    y = x[2:, 1::2]
    for idx in [(0, 0), (0, 1), (1, 2), (3, 2)]:
        assert address_from_storage_base(y, idx) == address_from_tensor_data_ptr(y, idx)


def test_tensor_address_info_cpu() -> None:
    x = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    info = tensor_address_info(x)
    assert info.shape == (3, 4)
    assert info.stride == (4, 1)
    assert info.element_size_bytes == 2
    assert info.logical_nbytes == 24

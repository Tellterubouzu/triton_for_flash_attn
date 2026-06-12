from __future__ import annotations

from triton_flash_course.gpu_model import (
    describe_1d_launch,
    occupancy_sketch,
    summarize_instruction_text,
)


def test_describe_1d_launch_handles_tail() -> None:
    g = describe_1d_launch(35, 8)
    assert g.num_programs == 5
    assert g.last_program_elements == 3


def test_occupancy_sketch_simple_cuda_like_props() -> None:
    props = {
        "warp_size": 32,
        "max_threads_per_multi_processor": 2048,
        "max_warps_per_multiprocessor": 64,
        "max_blocks_per_multiprocessor": 32,
        "multi_processor_count": 108,
    }
    s = occupancy_sketch(num_warps=4, device_properties=props, backend="cuda")
    assert s.requested_threads_per_program == 128
    assert s.resident_programs_by_threads == 16
    assert s.resident_programs_by_warps == 16
    assert s.resident_programs_upper_bound == 16
    assert s.theoretical_active_warps == 64


def test_instruction_summary_detects_mma_tokens() -> None:
    text = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32; ldmatrix.sync; cp.async;"
    summary = summarize_instruction_text(text)
    assert summary.mma_like >= 1
    assert summary.ldmatrix_like >= 1
    assert summary.cp_async_like >= 1

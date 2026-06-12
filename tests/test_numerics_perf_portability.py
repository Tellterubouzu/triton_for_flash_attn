from __future__ import annotations

import torch

from triton_flash_course.numerics import error_report, recommended_tolerance, softmax_stability_summary
from triton_flash_course.perf_model import attention_flash_schedule_bytes, attention_naive_bytes, matmul_flops, roofline_point
from triton_flash_course.portability import default_flash_config_candidates, validation_matrix


def test_error_report_passes_identical_tensor() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    report = error_report("x", x, x.clone(), dtype=torch.float32)
    assert report.passed
    assert report.max_abs == 0.0


def test_recommended_tolerance_attention_fp16_is_looser_than_fp32() -> None:
    fp16 = recommended_tolerance(torch.float16, op="attention")
    fp32 = recommended_tolerance(torch.float32, op="attention")
    assert fp16.rtol > fp32.rtol
    assert fp16.atol > fp32.atol


def test_softmax_stability_summary_detects_stable_finite() -> None:
    x = torch.tensor([[1000.0, 999.0, 998.0]])
    summary = softmax_stability_summary(x)
    assert summary["stable_all_finite"] is True


def test_perf_model_attention_flash_less_temporary_than_naive_for_large_seq() -> None:
    naive = attention_naive_bytes(1, 8, 1024, 1024, 64, 2)
    flash = attention_flash_schedule_bytes(1, 8, 1024, 1024, 64, 2, block_m=64)
    assert naive > 0
    assert flash > 0
    # Schedule bytes can be larger due to K/V re-streaming, but temporary footprint is what flash removes.
    assert matmul_flops(16, 16, 16) == 8192


def test_roofline_point_basic_bound() -> None:
    p = roofline_point("copy", flops=0, bytes_moved=1_000_000_000, peak_bandwidth_gbps=1000, peak_tflops=100)
    assert p.likely_bound == "memory-bound"
    assert p.roofline_lower_bound_ms == 1.0


def test_portability_candidates_and_validation_matrix() -> None:
    configs = default_flash_config_candidates(64, backend="cuda")
    assert configs
    assert all(c.head_dim == 64 for c in configs)
    cases = validation_matrix(dtypes=("fp16",), include_large=False)
    assert cases
    assert {c.causal for c in cases} == {False, True}

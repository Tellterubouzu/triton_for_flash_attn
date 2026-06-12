from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    note: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorReport:
    name: str
    dtype: str
    shape: tuple[int, ...]
    max_abs: float
    max_rel: float
    mean_abs: float
    rms: float
    max_expected_abs: float
    finite_actual: bool
    finite_expected: bool
    rtol: float
    atol: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def recommended_tolerance(dtype: torch.dtype, *, op: str = "generic") -> Tolerance:
    """Return a practical starting tolerance for educational kernel tests.

    These are not universal correctness criteria. They are intentionally loose for
    attention-like reductions in fp16/bf16, where order of operations and fp32
    accumulation policy can change the last few bits.
    """
    op = op.lower()
    if dtype == torch.float64:
        return Tolerance(1e-8, 1e-10, "fp64 reference/debug path")
    if dtype == torch.float32:
        if op in {"attention", "softmax", "layernorm", "reduction"}:
            return Tolerance(2e-4, 2e-5, "fp32 reduction path")
        return Tolerance(1e-4, 1e-5, "fp32 elementwise/matmul path")
    if dtype == torch.float16:
        if op in {"attention", "softmax", "layernorm", "reduction"}:
            return Tolerance(3e-2, 3e-2, "fp16 input with fp32 internal reductions")
        return Tolerance(2e-2, 2e-2, "fp16 path")
    if dtype == torch.bfloat16:
        if op in {"attention", "softmax", "layernorm", "reduction"}:
            return Tolerance(5e-2, 5e-2, "bf16 has fewer mantissa bits than fp16")
        return Tolerance(4e-2, 4e-2, "bf16 path")
    return Tolerance(1e-3, 1e-3, "fallback tolerance")


def error_report(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    op: str = "generic",
    rtol: float | None = None,
    atol: float | None = None,
) -> ErrorReport:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    dtype = dtype or actual.dtype
    tol = recommended_tolerance(dtype, op=op)
    rtol = tol.rtol if rtol is None else float(rtol)
    atol = tol.atol if atol is None else float(atol)

    a = actual.detach().float()
    e = expected.detach().float()
    diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-12)
    rel = diff / denom
    finite_actual = bool(torch.isfinite(a).all().item())
    finite_expected = bool(torch.isfinite(e).all().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float(rel.max().item()) if rel.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0
    max_expected_abs = float(e.abs().max().item()) if e.numel() else 0.0
    passed = bool(torch.allclose(a, e, rtol=rtol, atol=atol) and finite_actual and finite_expected)
    return ErrorReport(
        name=name,
        dtype=str(dtype).replace("torch.", ""),
        shape=tuple(int(x) for x in actual.shape),
        max_abs=max_abs,
        max_rel=max_rel,
        mean_abs=mean_abs,
        rms=rms,
        max_expected_abs=max_expected_abs,
        finite_actual=finite_actual,
        finite_expected=finite_expected,
        rtol=rtol,
        atol=atol,
        passed=passed,
    )


def compare_or_raise(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    op: str = "generic",
    rtol: float | None = None,
    atol: float | None = None,
) -> ErrorReport:
    report = error_report(name, actual, expected, dtype=dtype, op=op, rtol=rtol, atol=atol)
    if not report.passed:
        raise AssertionError(report.to_dict())
    return report


def softmax_stability_summary(x: torch.Tensor, dim: int = -1) -> dict[str, float | bool]:
    """Compare naive exp/sum softmax with max-subtracted stable softmax."""
    xf = x.detach().float()
    naive_num = torch.exp(xf)
    naive = naive_num / naive_num.sum(dim=dim, keepdim=True)
    shifted = xf - xf.max(dim=dim, keepdim=True).values
    stable_num = torch.exp(shifted)
    stable = stable_num / stable_num.sum(dim=dim, keepdim=True)
    diff = (stable - naive).abs()
    return {
        "naive_all_finite": bool(torch.isfinite(naive).all().item()),
        "stable_all_finite": bool(torch.isfinite(stable).all().item()),
        "max_abs_diff_where_finite": float(diff[torch.isfinite(diff)].max().item()) if torch.isfinite(diff).any() else math.inf,
        "stable_row_sum_max_error": float((stable.sum(dim=dim) - 1).abs().max().item()),
    }

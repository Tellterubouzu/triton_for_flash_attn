from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class LaunchGeometry:
    """A small, architecture-neutral description of how work is partitioned."""

    n_elements: int
    block_size: int
    num_programs: int
    elements_per_program: int
    last_program_elements: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class OccupancySketch:
    """Upper-bound occupancy estimate from public device properties.

    This is intentionally approximate. Real occupancy also depends on compiler
    register allocation, shared memory allocation, barriers, instruction mix,
    architecture-specific scheduling rules, and launch bounds. The value is still
    useful as a first-pass sanity check for Triton configs.
    """

    backend: str
    sm_count: int | None
    warp_size: int | None
    requested_warps_per_program: int
    requested_threads_per_program: int | None
    max_threads_per_sm: int | None
    max_blocks_or_programs_per_sm: int | None
    max_warps_per_sm: int | None
    resident_programs_by_threads: int | None
    resident_programs_by_blocks: int | None
    resident_programs_by_warps: int | None
    resident_programs_upper_bound: int | None
    theoretical_active_warps: int | None
    theoretical_occupancy_fraction: float | None
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InstructionSummary:
    """Counts target-specific instruction tokens in compiler output text."""

    mma_like: int
    wgmma_like: int
    dot_like: int
    mfma_like: int
    ldmatrix_like: int
    cp_async_like: int
    tma_like: int
    load_like: int
    store_like: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("b must be positive")
    return (a + b - 1) // b


def describe_1d_launch(n_elements: int, block_size: int) -> LaunchGeometry:
    if n_elements < 0:
        raise ValueError("n_elements must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_programs = ceil_div(n_elements, block_size) if n_elements else 0
    last = 0 if num_programs == 0 else n_elements - (num_programs - 1) * block_size
    return LaunchGeometry(
        n_elements=n_elements,
        block_size=block_size,
        num_programs=num_programs,
        elements_per_program=block_size,
        last_program_elements=last,
    )


def detect_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu-or-unavailable"
    try:
        version = getattr(torch.version, "hip", None)
        if version:
            return "rocm"
    except Exception:
        pass
    return "cuda"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def occupancy_sketch(
    *,
    num_warps: int,
    device_properties: Mapping[str, Any] | None = None,
    backend: str | None = None,
) -> OccupancySketch:
    """Estimate a rough resident-program upper bound for a Triton config.

    In Triton, `num_warps` is a launch meta-parameter for the number of warps
    assigned to a program instance. This function maps it to a CUDA-like mental
    model: threads_per_program = num_warps * warp_size.
    """
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")

    props = dict(device_properties or {})
    backend = backend or detect_backend()
    warp_size = _int_or_none(props.get("warp_size"))
    if warp_size is None:
        # CUDA/NVIDIA is 32. AMD wavefront size is often 64 on CDNA, but ROCm can
        # expose a different value through PyTorch. Unknown is better than wrong.
        warp_size = 32 if backend == "cuda" else None

    max_threads_per_sm = _int_or_none(props.get("max_threads_per_multi_processor"))
    sm_count = _int_or_none(props.get("multi_processor_count"))

    # PyTorch exposes different names across builds. These are best-effort.
    max_blocks = _int_or_none(
        props.get("max_blocks_per_multiprocessor")
        or props.get("max_blocks_per_multi_processor")
        or props.get("max_ctas_per_multiprocessor")
    )

    max_warps_per_sm = _int_or_none(props.get("max_warps_per_multiprocessor"))
    if max_warps_per_sm is None and max_threads_per_sm is not None and warp_size:
        max_warps_per_sm = max_threads_per_sm // warp_size

    requested_threads = None if warp_size is None else num_warps * warp_size
    by_threads = None
    if requested_threads and max_threads_per_sm:
        by_threads = max_threads_per_sm // requested_threads

    by_warps = None
    if max_warps_per_sm is not None:
        by_warps = max_warps_per_sm // num_warps

    bounds = [x for x in [by_threads, by_warps, max_blocks] if x is not None]
    upper = min(bounds) if bounds else None

    active_warps = None if upper is None else upper * num_warps
    occupancy = None
    if active_warps is not None and max_warps_per_sm:
        occupancy = active_warps / max_warps_per_sm

    return OccupancySketch(
        backend=backend,
        sm_count=sm_count,
        warp_size=warp_size,
        requested_warps_per_program=num_warps,
        requested_threads_per_program=requested_threads,
        max_threads_per_sm=max_threads_per_sm,
        max_blocks_or_programs_per_sm=max_blocks,
        max_warps_per_sm=max_warps_per_sm,
        resident_programs_by_threads=by_threads,
        resident_programs_by_blocks=max_blocks,
        resident_programs_by_warps=by_warps,
        resident_programs_upper_bound=upper,
        theoretical_active_warps=active_warps,
        theoretical_occupancy_fraction=occupancy,
        caveat=(
            "Upper bound only. Actual occupancy can be lower due to registers, "
            "shared memory, barriers, compiler scheduling, and architecture rules."
        ),
    )


def summarize_instruction_text(text: str) -> InstructionSummary:
    lowered = text.lower()
    return InstructionSummary(
        mma_like=len(re.findall(r"\bmma(?:\.|_)|\bwmma(?:\.|_)", lowered)),
        wgmma_like=len(re.findall(r"\bwgmma(?:\.|_)", lowered)),
        dot_like=len(re.findall(r"\bdot(?:\.|_|\b)", lowered)),
        mfma_like=len(re.findall(r"\b(v_)?mfma", lowered)),
        ldmatrix_like=len(re.findall(r"\bldmatrix", lowered)),
        cp_async_like=len(re.findall(r"\bcp\.async|async_copy", lowered)),
        tma_like=len(re.findall(r"\btma\b|tensor_memory", lowered)),
        load_like=len(re.findall(r"\bld(?:\.|_|\b)|global_load|flat_load", lowered)),
        store_like=len(re.findall(r"\bst(?:\.|_|\b)|global_store|flat_store", lowered)),
    )


def summarize_compiled_asm(asm: Mapping[str, Any]) -> dict[str, InstructionSummary]:
    out: dict[str, InstructionSummary] = {}
    for key, value in asm.items():
        if isinstance(value, bytes):
            try:
                text = value.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        else:
            text = str(value)
        out[str(key)] = summarize_instruction_text(text)
    return out


def pretty_print_mapping_table() -> str:
    """Return a compact CUDA-vs-Triton vocabulary table."""
    rows = [
        ("GPU function", "kernel", "@triton.jit function", "GPU 上で実行される関数"),
        ("launch", "kernel<<<grid, block>>>", "kernel[grid](...)", "CPU が GPU に仕事を投入する"),
        ("work unit", "thread block / CTA", "program instance", "1 つの tile を担当する実行単位"),
        ("id", "blockIdx/threadIdx", "tl.program_id + tl.arange", "担当データの offset を作る"),
        ("SIMT group", "warp", "num_warps meta-parameter", "実際の thread 束は Triton が割り当てる"),
        ("matrix unit", "Tensor Core / MFMA", "tl.dot", "条件が合うと専用行列演算へ lower される"),
        ("on-chip storage", "register/shared memory", "compiler-managed block values", "Triton では明示配列ではなく compiler 管理が多い"),
    ]
    width = [max(len(r[i]) for r in rows + [("Concept", "CUDA", "Triton", "Meaning")]) for i in range(4)]
    header = ("Concept", "CUDA", "Triton", "Meaning")
    lines = [
        " | ".join(header[i].ljust(width[i]) for i in range(4)),
        "-+-".join("-" * width[i] for i in range(4)),
    ]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(width[i]) for i in range(4)))
    return "\n".join(lines)

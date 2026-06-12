from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RooflinePoint:
    name: str
    flops: int
    bytes_moved: int
    arithmetic_intensity_flop_per_byte: float
    memory_lower_bound_ms: float | None
    compute_lower_bound_ms: float | None
    roofline_lower_bound_ms: float | None
    likely_bound: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def roofline_point(
    name: str,
    *,
    flops: int,
    bytes_moved: int,
    peak_bandwidth_gbps: float | None = None,
    peak_tflops: float | None = None,
) -> RooflinePoint:
    intensity = float("inf") if bytes_moved == 0 else flops / bytes_moved
    mem_ms = None
    if peak_bandwidth_gbps and peak_bandwidth_gbps > 0:
        mem_ms = bytes_moved / (peak_bandwidth_gbps * 1e9) * 1e3
    comp_ms = None
    if peak_tflops and peak_tflops > 0:
        comp_ms = flops / (peak_tflops * 1e12) * 1e3
    if mem_ms is None and comp_ms is None:
        lower = None
        bound = "unknown"
    elif mem_ms is None:
        lower = comp_ms
        bound = "compute-estimate-only"
    elif comp_ms is None:
        lower = mem_ms
        bound = "memory-estimate-only"
    else:
        lower = max(mem_ms, comp_ms)
        bound = "memory-bound" if mem_ms >= comp_ms else "compute-bound"
    return RooflinePoint(name, flops, bytes_moved, intensity, mem_ms, comp_ms, lower, bound)


def elementwise_bytes(numel: int, element_size: int, *, n_reads: int, n_writes: int) -> int:
    return int(numel * element_size * (n_reads + n_writes))


def row_softmax_bytes(rows: int, cols: int, element_size: int, *, materialized_intermediates: bool) -> int:
    base = rows * cols * element_size
    if materialized_intermediates:
        # x read, max/temp writes/reads, exp writes/reads, y write. This is a rough lower bound.
        return 6 * base
    # Fused path: read row once and store output once; real kernels may reread or spill.
    return 2 * base


def matmul_flops(m: int, n: int, k: int) -> int:
    return int(2 * m * n * k)


def matmul_bytes(m: int, n: int, k: int, element_size: int) -> int:
    return int((m * k + k * n + m * n) * element_size)


def attention_flops(batch: int, heads: int, seq_q: int, seq_k: int, dim: int) -> int:
    # QK^T and PV. Softmax, mask, scale, exp are excluded.
    return int(4 * batch * heads * seq_q * seq_k * dim)


def attention_naive_bytes(batch: int, heads: int, seq_q: int, seq_k: int, dim: int, element_size: int) -> int:
    qkv_o = batch * heads * (seq_q * dim + 2 * seq_k * dim + seq_q * dim) * element_size
    scores_probs = 2 * batch * heads * seq_q * seq_k * element_size
    return int(qkv_o + scores_probs)


def attention_flash_schedule_bytes(
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    dim: int,
    element_size: int,
    *,
    block_m: int,
) -> int:
    # Simplified: Q and O once, K/V streamed for each query block.
    num_q_blocks = (seq_q + block_m - 1) // block_m
    q_o = batch * heads * 2 * seq_q * dim * element_size
    kv_stream = batch * heads * num_q_blocks * 2 * seq_k * dim * element_size
    return int(q_o + kv_stream)

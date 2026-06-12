from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class FlashConfigCandidate:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    head_dim: int
    backend_hint: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationCase:
    batch: int
    heads: int
    seq: int
    dim: int
    dtype: str
    causal: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def backend_from_torch() -> str:
    if not torch.cuda.is_available():
        return "cpu-or-unavailable"
    if getattr(torch.version, "hip", None):
        return "rocm"
    return "cuda"


def default_flash_config_candidates(head_dim: int, *, backend: str = "generic") -> list[FlashConfigCandidate]:
    if head_dim not in {16, 32, 64, 128}:
        return [
            FlashConfigCandidate(
                block_m=16,
                block_n=32,
                num_warps=4,
                num_stages=3,
                head_dim=head_dim,
                backend_hint=backend,
                reason="non-standard head_dim: start conservative and validate correctness first",
            )
        ]
    out: list[FlashConfigCandidate] = []
    if head_dim <= 32:
        specs = [(32, 64, 4, 3), (64, 64, 4, 3), (32, 128, 4, 4)]
    elif head_dim == 64:
        specs = [(32, 64, 4, 3), (64, 64, 4, 3), (64, 128, 8, 4)]
    else:
        specs = [(16, 64, 4, 3), (32, 64, 4, 3), (32, 128, 8, 4)]
    for block_m, block_n, num_warps, num_stages in specs:
        out.append(
            FlashConfigCandidate(
                block_m=block_m,
                block_n=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
                head_dim=head_dim,
                backend_hint=backend,
                reason="small portable sweep candidate; benchmark on target GPU",
            )
        )
    return out


def validation_matrix(
    *,
    dtypes: Iterable[str] = ("fp16", "bf16"),
    include_large: bool = False,
) -> list[ValidationCase]:
    seqs = [1, 7, 128, 257, 512]
    if include_large:
        seqs.extend([1024, 2048, 4096])
    dims = [16, 32, 64, 128]
    cases: list[ValidationCase] = []
    for dtype in dtypes:
        for causal in [False, True]:
            for seq in seqs:
                for dim in dims:
                    heads = 4 if dim <= 64 else 2
                    cases.append(ValidationCase(batch=1, heads=heads, seq=seq, dim=dim, dtype=dtype, causal=causal))
    return cases

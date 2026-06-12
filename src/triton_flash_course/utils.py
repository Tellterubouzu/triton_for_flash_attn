from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch


def has_triton_package() -> bool:
    return importlib.util.find_spec("triton") is not None


def has_cuda_or_rocm_device() -> bool:
    # torch.cuda is also used for ROCm builds.
    return torch.cuda.is_available()


def is_gpu_ready() -> bool:
    return has_triton_package() and has_cuda_or_rocm_device()


def require_gpu() -> None:
    if not is_gpu_ready():
        raise RuntimeError(
            "This lesson requires a CUDA/ROCm GPU and the triton package. "
            "Install Triton and run on a supported GPU."
        )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def next_power_of_2(x: int) -> int:
    if x < 1:
        return 1
    return 1 << (x - 1).bit_length()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    ms: float
    gbps: float | None = None
    tflops: float | None = None

    def as_row(self) -> list[str]:
        gbps = "-" if self.gbps is None else f"{self.gbps:.1f}"
        tflops = "-" if self.tflops is None else f"{self.tflops:.2f}"
        return [self.name, f"{self.ms:.4f}", gbps, tflops]


def benchmark_ms(fn: Callable[[], object], *, warmup: int = 25, rep: int = 100) -> float:
    """Benchmark a callable and return milliseconds.

    Uses CUDA events on GPU. Falls back to wall-clock time on CPU so lessons can be
    inspected without a GPU. The CPU fallback is not meant for kernel comparison.
    """
    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / rep

    for _ in range(max(1, warmup // 10)):
        fn()
    start_time = time.perf_counter()
    for _ in range(max(1, rep // 10)):
        fn()
    elapsed = time.perf_counter() - start_time
    return elapsed * 1000.0 / max(1, rep // 10)


def gbps(bytes_moved: int, ms: float) -> float:
    return bytes_moved / (ms * 1e-3) / 1e9


def tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def print_device_summary() -> None:
    print("Torch:", torch.__version__)
    print("CUDA/ROCm available:", torch.cuda.is_available())
    print("Triton package available:", has_triton_package())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        idx = torch.cuda.current_device()
        print("Current device:", torch.cuda.get_device_name(idx))
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        print(f"Total memory: {total_gb:.2f} GiB")
        if hasattr(props, "major") and hasattr(props, "minor"):
            print(f"Compute capability: {props.major}.{props.minor}")


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-5
    elif dtype == torch.float16:
        rtol, atol = 3e-2, 3e-2
    elif dtype == torch.bfloat16:
        rtol, atol = 4e-2, 4e-2
    else:
        rtol, atol = 1e-3, 1e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=f"{name} mismatch")


def make_qkv(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    scale = 1.0 / math.sqrt(dim)
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, generator=g) * scale
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, generator=g) * scale
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, generator=g)
    return q.contiguous(), k.contiguous(), v.contiguous()

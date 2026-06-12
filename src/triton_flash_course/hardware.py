from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch


MEMORY_PROP_CANDIDATES = [
    "total_memory",
    "shared_memory_per_block",
    "shared_memory_per_block_optin",
    "shared_memory_per_multiprocessor",
    "regs_per_block",
    "regs_per_multiprocessor",
    "l2_cache_size",
    "multi_processor_count",
    "warp_size",
    "max_threads_per_block",
    "max_threads_per_multi_processor",
    "major",
    "minor",
]


@dataclass(frozen=True)
class TensorAddressInfo:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    element_size_bytes: int
    numel: int
    logical_nbytes: int
    storage_nbytes: int
    storage_offset_elements: int
    tensor_data_ptr: int
    storage_data_ptr: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceReport:
    cuda_available: bool
    current_device: int | None
    name: str | None
    total_hbm_bytes: int | None
    free_hbm_bytes: int | None
    used_hbm_bytes: int | None
    device_properties: dict[str, Any]
    nvidia_smi: dict[str, str] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def bytes_to_gib(x: int | None) -> float | None:
    if x is None:
        return None
    return x / 1024**3


def human_bytes(x: int | None) -> str:
    if x is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(x)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{x} B"


def _safe_getattr(obj: object, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def get_device_properties_dict(device: int | torch.device | str | None = None) -> dict[str, Any]:
    """Return a stable subset of torch.cuda device properties plus any public scalar fields.

    PyTorch exposes a CUDA-device-properties object whose attributes differ across
    CUDA/ROCm versions and GPU architectures. This function avoids relying on one
    exact schema and records both common fields and simple scalar extras.
    """
    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(device)
    out: dict[str, Any] = {}
    for key in MEMORY_PROP_CANDIDATES:
        value = _safe_getattr(props, key)
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    for key in dir(props):
        if key.startswith("_") or key in out:
            continue
        try:
            value = getattr(props, key)
        except Exception:
            continue
        if isinstance(value, (int, float, str, bool)):
            out[key] = value
    return out


def get_hbm_free_total(device: int | torch.device | str | None = None) -> tuple[int | None, int | None]:
    """Return (free_bytes, total_bytes) for device-wide HBM/global memory.

    This uses torch.cuda.mem_get_info when available. It is device-wide driver
    state, not the number of bytes currently owned by a specific tensor or by the
    PyTorch caching allocator.
    """
    if not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info(device)
        return int(free), int(total)
    except Exception:
        props = torch.cuda.get_device_properties(device)
        return None, int(props.total_memory)


def query_nvidia_smi(device_index: int = 0) -> dict[str, str] | None:
    """Best-effort nvidia-smi query.

    This intentionally remains optional so AMD/ROCm and minimal containers still
    work. For ROCm, use rocminfo/rocm-smi manually; PyTorch still exposes many
    properties through torch.cuda on ROCm builds.
    """
    exe = os.environ.get("NVIDIA_SMI", "nvidia-smi")
    query = (
        "name,memory.total,memory.free,memory.used,clocks.current.graphics,"
        "clocks.current.memory,temperature.gpu,utilization.gpu,utilization.memory"
    )
    cmd = [
        exe,
        f"--id={device_index}",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except Exception:
        return None
    if not raw:
        return None
    keys = query.split(",")
    values = [x.strip() for x in raw.split(",")]
    return dict(zip(keys, values, strict=False))


def get_device_report(device: int | torch.device | str | None = None) -> DeviceReport:
    if not torch.cuda.is_available():
        return DeviceReport(
            cuda_available=False,
            current_device=None,
            name=None,
            total_hbm_bytes=None,
            free_hbm_bytes=None,
            used_hbm_bytes=None,
            device_properties={},
            nvidia_smi=None,
        )
    idx = torch.cuda.current_device() if device is None else int(torch.device(device).index or 0) if not isinstance(device, int) else device
    free, total = get_hbm_free_total(idx)
    used = None if free is None or total is None else total - free
    return DeviceReport(
        cuda_available=True,
        current_device=idx,
        name=torch.cuda.get_device_name(idx),
        total_hbm_bytes=total,
        free_hbm_bytes=free,
        used_hbm_bytes=used,
        device_properties=get_device_properties_dict(idx),
        nvidia_smi=query_nvidia_smi(idx),
    )


def print_device_report(device: int | torch.device | str | None = None) -> None:
    report = get_device_report(device)
    print(report.to_json())
    if report.cuda_available:
        print("\nReadable summary")
        print(f"  GPU             : {report.name}")
        print(f"  HBM total       : {human_bytes(report.total_hbm_bytes)}")
        print(f"  HBM free        : {human_bytes(report.free_hbm_bytes)}")
        print(f"  HBM used        : {human_bytes(report.used_hbm_bytes)}")
        props = report.device_properties
        for key in [
            "multi_processor_count",
            "warp_size",
            "shared_memory_per_block",
            "shared_memory_per_multiprocessor",
            "regs_per_block",
            "regs_per_multiprocessor",
            "l2_cache_size",
            "max_threads_per_block",
        ]:
            if key in props and props[key] is not None:
                value = props[key]
                if "memory" in key or "cache" in key:
                    value = human_bytes(int(value))
                print(f"  {key:28s}: {value}")


def tensor_address_info(t: torch.Tensor) -> TensorAddressInfo:
    storage = t.untyped_storage()
    return TensorAddressInfo(
        shape=tuple(int(x) for x in t.shape),
        stride=tuple(int(x) for x in t.stride()),
        dtype=str(t.dtype).replace("torch.", ""),
        device=str(t.device),
        element_size_bytes=t.element_size(),
        numel=t.numel(),
        logical_nbytes=t.numel() * t.element_size(),
        storage_nbytes=storage.nbytes(),
        storage_offset_elements=t.storage_offset(),
        tensor_data_ptr=t.data_ptr(),
        storage_data_ptr=storage.data_ptr(),
    )


def linear_offset_elements(index: Iterable[int], stride: Iterable[int], storage_offset: int = 0) -> int:
    """Compute storage element offset from a logical index and element strides."""
    return int(storage_offset + sum(int(i) * int(s) for i, s in zip(index, stride, strict=True)))


def address_from_storage_base(t: torch.Tensor, index: Iterable[int]) -> int:
    """Compute byte address using storage base pointer + storage_offset + strides.

    This is the most explicit formula for PyTorch views.
    """
    off = linear_offset_elements(index, t.stride(), t.storage_offset())
    return t.untyped_storage().data_ptr() + off * t.element_size()


def address_from_tensor_data_ptr(t: torch.Tensor, index: Iterable[int]) -> int:
    """Compute byte address using tensor.data_ptr() as the first logical element.

    tensor.data_ptr() already includes storage_offset(). Therefore this formula
    must not add storage_offset again.
    """
    off = linear_offset_elements(index, t.stride(), 0)
    return t.data_ptr() + off * t.element_size()


def print_tensor_address_table(t: torch.Tensor, indices: Iterable[tuple[int, ...]]) -> None:
    info = tensor_address_info(t)
    print(json.dumps(info.to_dict(), indent=2, ensure_ascii=False))
    print("\nindex -> element_offset -> byte_address")
    for idx in indices:
        off = linear_offset_elements(idx, t.stride(), t.storage_offset())
        addr = address_from_storage_base(t, idx)
        print(f"  {idx}: offset={off}, address=0x{addr:x}")


def estimate_tensor_read_write_bytes(*, reads: Iterable[torch.Tensor], writes: Iterable[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in reads) + sum(t.numel() * t.element_size() for t in writes)


def allocator_snapshot() -> dict[str, Any]:
    """Return a small, stable view of the PyTorch CUDA caching allocator."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "memory_allocated": int(torch.cuda.memory_allocated()),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "memory_reserved": int(torch.cuda.memory_reserved()),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
    }

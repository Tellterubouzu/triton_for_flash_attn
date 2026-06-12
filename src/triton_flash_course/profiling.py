from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler


def profile_callable(
    fn: Callable[[], object],
    *,
    name: str = "region",
    trace_dir: str | Path | None = None,
    warmup: int = 3,
    active: int = 5,
    record_shapes: bool = True,
    profile_memory: bool = True,
) -> str:
    """Run torch.profiler and return a text table sorted by CUDA time if available."""
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    kwargs = {}
    if trace_dir is not None:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        kwargs["on_trace_ready"] = tensorboard_trace_handler(str(trace_dir))

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=False,
        **kwargs,
    ) as prof:
        for _ in range(active):
            with record_function(name):
                fn()
            prof.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    return prof.key_averages(group_by_input_shape=True).table(sort_by=sort_by, row_limit=25)


@contextlib.contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from triton_flash_course.utils import benchmark_ms, get_device, print_device_summary


def main() -> None:
    print_device_summary()
    device = get_device()
    x = torch.randn(1_000_000, device=device)
    y = torch.randn_like(x)
    ms = benchmark_ms(lambda: x + y, warmup=5, rep=20)
    print(f"baseline torch add on {device}: {ms:.4f} ms")
    print("次: lessons/01_gpu_execution_model.py を実行してください。")


if __name__ == "__main__":
    main()

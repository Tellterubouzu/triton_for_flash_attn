from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from triton_flash_course.hardware import print_device_report


if __name__ == "__main__":
    print("GPU device / memory hierarchy report")
    print_device_report()
    print("\nNotes:")
    print("- HBM total/free comes from torch.cuda.mem_get_info when available.")
    print("- shared_memory_per_block is an upper bound exposed by the runtime, not a guarantee that a Triton block tensor is physically placed there.")
    print("- For L2 hit rate, register spill, and warp stall, use Nsight Compute or rocprof.")

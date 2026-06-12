# Profiler Metrics: PyTorch Profiler, Nsight, rocprof

## 目的

PyTorch profiler は「どの operator / kernel が高いか」を見つける道具です。Nsight Compute や rocprof は「その kernel がなぜ遅いか」を見る道具です。Triton kernel を改善するときは、この 2 段階を分けます。

## PyTorch profiler で見るもの

- operator-level の self time / total time
- CUDA/HIP kernel-level の時間
- shape ごとの operator grouping
- memory allocation / deallocation
- trace timeline

教材では `src/triton_flash_course/profiling.py` と `benchmarks/profile_attention.py` を使います。

```bash
python benchmarks/profile_attention.py --trace-dir traces --seq 2048 --dim 64 --dtype fp16
```

## Nsight Compute / rocprof で見るもの

NVIDIA では Nsight Compute、AMD では rocprof / rocprof-compute を使います。見るべき metric は次です。

| 分類 | 観点 |
|---|---|
| Memory | achieved HBM bandwidth, L2 hit rate, global load/store transaction, sector waste |
| Compute | Tensor Core / MFMA utilization, FP32 pipe usage, exp/reduction cost |
| Scheduling | occupancy, eligible warps per cycle, long scoreboard stall, barrier stall |
| Resource | register usage, local memory spill, shared memory usage |
| Attention-specific | BLOCK_M/BLOCK_N sweep, causal boundary block, K/V reread cost |

command template は次で確認できます。

```bash
python lessons/27_ncu_rocprof_workflow.py
```

## 注意

- token search で `mma` や `mfma` が見えることは、Tensor Core/MFMA が使われた可能性を示しますが、utilization までは保証しません。
- occupancy が高い kernel が必ず速いわけではありません。register pressure を下げて occupancy を上げても、memory locality や Tensor Core 利用率が落ちれば遅くなります。
- small shape では kernel launch overhead が支配的になるため、large shape の roofline と分けて評価します。

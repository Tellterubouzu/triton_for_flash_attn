# Production FlashAttention Checklist

## correctness

- forward: dtype, shape, causal, non-power-of-two, edge case を validation matrix 化する。
- backward: `torch.autograd.gradcheck`、finite difference、実モデル loss で確認する。
- special value: NaN/Inf の扱いを仕様化する。
- determinism: deterministic mode が必要かを決める。

## performance

- warmup と measurement を分ける。
- CUDA/HIP event timing と profiler timing を分ける。
- small sequence と large sequence を分ける。
- `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` を GPU ごとに sweep する。
- HBM bandwidth, L2 hit rate, register spill, tensor pipe utilization を確認する。

## portability

- CUDA と ROCm の両方で correctness CI を持つ。
- version: torch / triton / driver / CUDA / ROCm を保存する。
- backend-specific config は分離し、core kernel には vendor 固有 inline assembly を入れない。
- fallback path を必ず用意する。

## integration

- `torch.compile` で graph break しないか確認する。
- training では autograd wrapper または `torch.library` integration を設計する。
- inference では KV cache layout と paged attention との接続を設計する。
- benchmark は model end-to-end と kernel microbenchmark の両方を取る。

```bash
python lessons/28_production_kernel_checklist.py
```

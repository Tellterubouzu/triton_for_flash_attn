# FlashAttention Validation and Portability

## correctness before performance

FlashAttention kernel は、単一 shape で PyTorch と近い結果が出ても不十分です。non-power-of-two sequence、small sequence、causal boundary、head_dim の違いで壊れやすいため、validation matrix を先に固定します。

```bash
python lessons/25_flash_attention_validation_matrix.py --max-cases 64
```

## portable default と tuned config

portable default は「多くの GPU で失敗せず、極端に遅くない config」です。tuned config は「特定 GPU / dtype / shape で最速だった config」です。両者は分けて管理します。

よく sweep する軸は次です。

- `BLOCK_M`: query block の row 数
- `BLOCK_N`: key/value block の column 数
- `num_warps`: 1 program instance に割り当てる warp 数
- `num_stages`: software pipelining の段数
- `head_dim`: 16 / 32 / 64 / 128
- causal / non-causal
- backend: CUDA / ROCm

教材の `src/triton_flash_course/portability.py` は、小さい candidate list と validation case を返します。

```bash
python lessons/24_flash_attention_autotune_portability.py --seq 1024 --dim 64 --dtype fp16
```

## NVIDIA と AMD の違い

Triton のコードを可能な限り共通にしても、warp/wavefront、MMA/MFMA、LDS/shared memory、compiler lowering、available dtype は異なります。したがって、config は backend ごとに benchmark して記録します。

## 保存すべき artifact

- GPU 名、driver、CUDA/ROCm、PyTorch、Triton version
- shape/dtype/causal/head_dim
- best config と候補 config
- correctness report
- benchmark median / p10 / p90
- profiler summary
- generated IR/assembly の key token summary

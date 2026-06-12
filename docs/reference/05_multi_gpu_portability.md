# 05. 複数 GPU で使える kernel にするための設計

「いろんな GPU で使える」とは、単一 GPU で最高速を出すことではありません。最低限、正しく動き、極端に遅くならず、GPU ごとに config を選べる状態を指します。

## Portability の基本方針

1. vendor 固有の inline assembly を避ける。
2. `tl.dot`, `tl.load`, `tl.store`, `tl.where` など Triton の標準 primitive を中心に書く。
3. config を hard-code せず、shape と GPU ごとに sweep / autotune できるようにする。
4. correctness tests を NVIDIA / AMD の両方で走らせる。
5. production では CI matrix に GPU 世代、driver、ROCm/CUDA、Triton version を含める。

## shape policy

この教材の FlashAttention kernel は次を主対象にしています。

- layout: `[batch, heads, seq, head_dim]`
- dtype: fp16 / bf16
- head_dim: 16, 32, 64, 128
- causal: true / false

head_dim > 128、variable length、GQA/MQA、paged KV cache は発展課題です。

## config の見方

- `BLOCK_M`: 1 program が担当する query 数。大きいほど Q reuse は増えますが register pressure も増えます。
- `BLOCK_N`: 1 回に読む key/value 数。大きいほど K/V read の効率が上がる可能性がありますが、tile が重くなります。
- `BLOCK_D`: head_dim の padded power-of-two。境界 mask が必要です。
- `num_warps`: 並列度。大きい tile ほど増やす候補になります。
- `num_stages`: pipeline stages。GPU や memory latency により最適値が変わります。

## 実験ログの推奨形式

```text
GPU: A100-SXM4-80GB
Backend: CUDA
Torch: 2.x
Triton: 3.x
Shape: B=1,H=8,N=2048,D=64,dtype=fp16,causal=False
Config: BLOCK_M=16,BLOCK_N=64,num_warps=4,num_stages=3
Latency: ... ms
TFLOP/s: ...
Correctness: max_abs=..., max_rel=...
```

## よくある落とし穴

- A100 で速い config が RTX 4090 や MI300 で速いとは限りません。
- compile cache の cold start と steady-state latency を混ぜてはいけません。
- `torch.nn.functional.scaled_dot_product_attention` は環境により backend が変わるため、単純な naive PyTorch との比較も残します。
- dtype tolerance を厳しくしすぎると、softmax の計算順序差を誤って bug と判断します。

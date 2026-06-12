# 04. PyTorch 実装から kernel 化対象を見つける workflow

## Step 1: まず PyTorch baseline を正しく書く

最初から Triton に行かず、数式に最も近い PyTorch 実装を書きます。FlashAttention なら naive attention をあえて書きます。

```python
scores = q @ k.transpose(-2, -1) * scale
probs = torch.softmax(scores, dim=-1)
out = probs @ v
```

## Step 2: `torch.compile` を試す

単純な elementwise chain や小さい fusion は、手書き Triton より `torch.compile` の方が低コストです。graph break がある場合は、その場所を確認します。

## Step 3: profiler を見る

`torch.profiler` で見るべき項目です。

- self CUDA time / CUDA time total
- input shape
- memory allocation
- call count
- kernel 名

attention では、`matmul`, `bmm`, `softmax`, `scaled_dot_product_attention`、および memory allocation が重要です。

## Step 4: kernel 化の判断

Triton 化に向く条件:

- 複数 operator の間に巨大な中間 tensor がある。
- elementwise / reduction / matmul を融合できる。
- PyTorch eager や `torch.compile` では graph break や dynamic control flow が残る。
- shape がある程度限定されており、kernel specialization が効く。

Triton 化に向かない条件:

- 単体 matmul で vendor library に任せればよい。
- CPU data loading が支配的。
- batch が小さすぎて launch overhead が支配的。
- shape が極端に多様で autotune cache が効かない。

## Step 5: correctness first

Triton kernel は速さより先に正しさを固定します。

- dtype ごとの tolerance を明示する。
- shape 境界、非 power-of-two、causal/non-causal を試す。
- NaN/Inf を含む入力は別テストにする。
- small shape で reference と比較する。

## Step 6: benchmark と regression

benchmark は最低限、次を記録します。

- GPU 名
- PyTorch/Triton version
- dtype
- shape
- config: `BLOCK_M`, `BLOCK_N`, `BLOCK_D`, `num_warps`, `num_stages`
- warmup/rep
- 平均 latency
- TFLOP/s または GB/s

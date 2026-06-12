# 02. Triton Programming Model

Triton は thread を直接書くのではなく、1 つの program が tile を処理するという抽象化で kernel を書きます。

## program_id

`tl.program_id(axis=0)` は、現在の program が grid のどの位置を担当しているかを表します。1D vector add では、program id が block index に対応します。

```python
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

## mask

Triton では `tl.load` / `tl.store` に mask を渡し、境界外 access を防ぎます。

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
tl.store(y_ptr + offsets, y, mask=mask)
```

FlashAttention では、mask は 3 種類あります。

1. sequence length の境界 mask
2. head_dim の境界 mask
3. causal attention の上三角 mask

## constexpr

`tl.constexpr` は compile-time constant です。`BLOCK_M`, `BLOCK_N`, `BLOCK_D`, `CAUSAL` のような値は、実行時の tensor data ではなく kernel specialization を決める値です。

## reduction

`tl.max(x, axis=0)` や `tl.sum(x, axis=0)` は、tile 内の reduction を表します。softmax と layernorm では row-wise reduction、FlashAttention では block 内 softmax 統計量の更新に使います。

## tl.dot

`tl.dot(a, b)` は tile matmul です。FlashAttention forward では次の 2 つに使います。

\[
S_{block}=Q_{block}K_{block}^\top
\]

\[
Acc_{block}=P_{block}V_{block}
\]

`tl.dot` の dtype、input precision、tile shape は性能に強く影響します。自作 kernel では `torch.matmul` 単体に勝つことより、softmax や scaling、mask、normalization と融合することを重視します。

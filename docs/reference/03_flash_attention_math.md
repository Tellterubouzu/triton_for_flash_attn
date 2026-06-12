# 03. FlashAttention の数式

この章は、FlashAttention forward kernel が何を計算しているかを数式で整理します。

## 通常の attention

この数式は、Query と Key の scaled dot-product を softmax で正規化し、Value の加重和を取る操作を表します。

\[
O=\mathrm{softmax}(S)V,\quad S=QK^\top/\sqrt{d}
\]

各 query 行 \(i\) については次の形です。

\[
O_i = \sum_j \frac{\exp(S_{ij})}{\sum_t \exp(S_{it})} V_j
\]

## 数値安定な softmax

この数式は、softmax の overflow を避けるため、各行の最大値を引いて計算する形を表します。

\[
P_{ij}=\frac{\exp(S_{ij}-m_i)}{\sum_t\exp(S_{it}-m_i)},\quad m_i=\max_j S_{ij}
\]

最大値を引いても softmax は変わりません。

## Online softmax

FlashAttention は Key/Value を block ごとに読むため、行全体の最大値と分母を一度に知ることができません。そこで、これまで見た block の統計量を保持します。

この数式は、古い最大値 \(m_{old}\)、古い正規化分母 \(l_{old}\)、新しい score block から、新しい統計量を更新する操作を表します。

\[
m_{new}=\max(m_{old}, \max_j S_j)
\]

\[
l_{new}=e^{m_{old}-m_{new}}l_{old}+\sum_j e^{S_j-m_{new}}
\]

出力 accumulator も同じ rescale が必要です。

\[
acc_{new}=e^{m_{old}-m_{new}}acc_{old}+\sum_j e^{S_j-m_{new}}V_j
\]

最後に次を保存します。

\[
O=acc/l
\]

## causal mask

causal attention では、query index \(i\) が key index \(j\) より小さい未来 token を見てはいけません。

\[
S_{ij}=-\infty\quad\mathrm{if}\quad j>i
\]

Triton kernel では次のように block 内で mask します。

```python
qk = tl.where(k_idx[None, :] <= q_idx[:, None], qk, -float("inf"))
```

## backward の入口

この数式は、attention backward で必要になる主要な勾配を表します。

\[
dV=P^\top dO
\]

\[
dP=dO V^\top
\]

\[
dS=P\odot(dP-\sum_j P_jdP_j)
\]

\[
dQ=dS K/\sqrt{d},\quad dK=dS^\top Q/\sqrt{d}
\]

forward と同様に、\(P\) を HBM に materialize せず、保存した log-sum-exp を使って block ごとに再計算します。

# 11. FlashAttention IO schedule at address/block level

この章の目的は、FlashAttention を「softmax の高速版」ではなく、HBM traffic を減らす schedule として読めるようにすることです。

## naive attention の memory schedule

この数式は scaled dot-product attention を表します。

\[
O = \mathrm{softmax}\left(QK^\top / \sqrt{d}\right)V
\]

naive 実装は次の中間を HBM に置きがちです。

- `S[B,H,N,N] = QK^T / sqrt(d)`
- `P[B,H,N,N] = softmax(S)`
- `O[B,H,N,D] = PV`

`N=4096`, `B=1`, `H=32`, fp16 なら、`S` だけで `1*32*4096*4096*2 ≈ 1 GiB` です。`P` も同程度です。forward の一時領域だけで巨大になります。

## tiled schedule

1 program は `BLOCK_M` 個の query rows を担当します。

- Q block: `[BLOCK_M, D]`
- K block: `[BLOCK_N, D]`
- V block: `[BLOCK_N, D]`
- score block: `[BLOCK_M, BLOCK_N]`
- accumulator: `[BLOCK_M, D]`
- row statistics: `m[BLOCK_M]`, `l[BLOCK_M]`

K/V を `BLOCK_N` ごとに HBM から stream し、score block をその場で使い捨てます。score/probability matrix は HBM に書きません。

## online softmax 更新

この式は、新しい K block を読んだ後の行ごとの最大値を更新します。

\[
m_{new}=\max(m_{old}, \max_j s_j)
\]

この式は、古い softmax 分母と新しい block の分母を、同じ最大値基準へ再スケールして足し合わせます。

\[
l_{new}=e^{m_{old}-m_{new}}l_{old}+\sum_j e^{s_j-m_{new}}
\]

この式は、出力 accumulator を同じ最大値基準へ再スケールし、新しい probability-weighted value を加えます。

\[
acc_{new}=e^{m_{old}-m_{new}}acc_{old}+\sum_j e^{s_j-m_{new}}v_j
\]

最後に次で出力します。

\[
o = acc / l
\]

## HBM bytes の概算

FlashAttention forward の最小読み書きは、かなり粗く見ると次です。

- Q: 各 query block ごとに一度読む。
- K/V: 各 query block が全 K/V block を読む。したがって batch/head ごとに `ceil(N/BLOCK_M)` 回 stream される。
- O: 一度書く。
- LSE: backward 用に保存するなら一度書く。

この教材の `lessons/22_flash_attention_io_accounting.py` は naive と flash の一時 memory、および概算 HBM traffic を shape ごとに出します。

## kernel design checklist

- Q/K/V/O の stride は想定通りか。
- head_dim は contiguous に読めるか。
- causal mask は score block 内で処理され、余計な HBM write を生んでいないか。
- `BLOCK_M/BLOCK_N` を大きくしすぎて register pressure を上げていないか。
- `num_stages` を増やしたときに latency hiding と resource usage のどちらが勝っているか。
- GPU ごとに default config と tuned config を分けて記録しているか。

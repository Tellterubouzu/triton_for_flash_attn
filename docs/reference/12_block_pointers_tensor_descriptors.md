# 12. Block pointers and tensor descriptors

この章の目的は、Triton の手書き pointer tensor と `tl.make_block_ptr` の違いを理解することです。

## explicit pointer tensor

初期の教材では次のように pointer tensor を手で作ります。

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
x = tl.load(ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
```

これは address 計算が見えやすく、学習の入口に向いています。

## block pointer

`tl.make_block_ptr` は、base pointer、親 tensor の shape、strides、block offset、block_shape、order をまとめて block descriptor にします。

```python
x_block = tl.make_block_ptr(
    base=x_ptr,
    shape=(M, N),
    strides=(N, 1),
    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)
x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
```

この形では `mask` を直接渡すのではなく、`boundary_check` と `padding_option` で端の tile を処理します。

## tl.advance

K/V を sequence 方向へ stream するときは、block pointer を次の tile へ進めると読みやすくなります。

```python
k_block = tl.make_block_ptr(... offsets=(0, d_start), block_shape=(BLOCK_N, BLOCK_D), ...)
for start_n in range(0, N_CTX, BLOCK_N):
    k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
    k_block = tl.advance(k_block, (BLOCK_N, 0))
```

FlashAttention の K/V loop はこの形に近いです。

## order の直感

`order` は block 内の memory layout / traversal に関する compiler hint です。contiguous dimension を最後に持つ row-major 2D tensor では、多くの場合 `order=(1, 0)` を使います。GPU 世代や TMA 対応など backend 依存の意味もあるため、単なる Python の view 変換ではありません。

## 注意

block pointer は address 計算を隠すので、最初から使うと stride/storage offset の理解が浅くなります。この教材では explicit pointer tensor で address を理解してから、FlashAttention に近い block pointer へ移行します。

## 実験

```bash
python lessons/10_block_pointers_boundary_check.py --m 1025 --n 2049 --dtype fp16
```

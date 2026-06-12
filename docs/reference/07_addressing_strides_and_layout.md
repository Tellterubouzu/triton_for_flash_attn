# 07. Addressing, strides, storage offset

この章の目的は、PyTorch Tensor の index が GPU kernel 内の pointer arithmetic にどう変換されるかを理解することです。

## アドレス式

次の式は、N 次元 tensor の論理 index から storage 上の byte address を求める式です。

\[
\mathrm{addr}(i_0,\dots,i_{n-1})
=
\mathrm{storage\_base}
+
\left(
\mathrm{storage\_offset}
+
\sum_{d=0}^{n-1} i_d s_d
\right)
\cdot \mathrm{element\_size}
\]

ここで `s_d` は PyTorch の `tensor.stride()[d]` です。stride は byte ではなく element 単位です。

Triton の `ptr + offsets` も通常は element 単位の pointer arithmetic です。`float16*` に `+1` すると 2 byte 進み、`float32*` に `+1` すると 4 byte 進む、という意味です。

## data_ptr と storage base

PyTorch には少なくとも 2 種類の pointer 観察点があります。

- `tensor.untyped_storage().data_ptr()`: storage の先頭 byte address。
- `tensor.data_ptr()`: その Tensor の最初の論理要素の byte address。view では `storage_offset` を反映済みです。

したがって、address を計算するときは次のどちらかを使います。

```python
# storage base から計算する場合
addr = storage.data_ptr() + (tensor.storage_offset() + sum(i_d * stride_d)) * element_size

# tensor.data_ptr() から計算する場合
addr = tensor.data_ptr() + sum(i_d * stride_d) * element_size
```

`storage_offset` を二重に足すバグは、custom kernel で view を扱うときに非常に起きやすいです。

## contiguous と coalescing

contiguous な `[B, H, N, D]` tensor の最後の次元 `D` を warp lane が連続的に読むと、memory transaction は効率的になりやすいです。一方、stride の大きい次元を lane ごとに読むと、各 lane が離れた cache line を触り、実効 bandwidth が落ちます。

FlashAttention では、head_dim 方向を contiguous に読める layout にして、Q/K/V block を `[BLOCK_M, BLOCK_D]` または `[BLOCK_N, BLOCK_D]` として読むのが基本です。

## 実験

```bash
python lessons/04_tensor_addresses_strides_layouts.py
python lessons/06_coalescing_strides_cache.py --numel 67108864 --dtype fp16
```

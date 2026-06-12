# Module 04: Tensor address、storage_offset、stride、layout

## 位置づけ

PyTorch Tensor の論理 index が実際の byte address に変換される式を理解し、Triton pointer arithmetic に対応づける。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| pointer | memory 上の位置を指す値。Triton の `ptr + offsets` は要素単位の pointer arithmetic として扱う。 |
| byte address | byte 単位で数えた memory address。 |
| data_ptr | PyTorch Tensor の先頭要素付近を指す pointer 値を取得する API。 |
| storage | Tensor の実データを保持する memory 領域。view は同じ storage を共有できる。 |
| storage_offset | Tensor の論理 index 0 が storage の何要素目から始まるかを表す offset。 |
| stride | 各次元で index を 1 増やしたときに進む element 数。byte 数ではない。 |
| contiguous | 標準的な row-major layout で隙間なく並んでいる状態。 |

## 数式

この数式は、N 次元 Tensor の論理 index から実際の byte address を求める式です。stride は byte 単位ではなく element 単位である点が重要です。

\[
\mathrm{addr}(i_0,\dots,i_{n-1})
=
\mathrm{storage\_base}
+
\left(\mathrm{storage\_offset}+\sum_{d=0}^{n-1} i_d s_d\right)
\cdot \mathrm{element\_size}
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/04_tensor_addresses_strides_layouts.md`](../../exercises/04_tensor_addresses_strides_layouts.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/04_tensor_addresses_strides_layouts.py`](../../lessons/04_tensor_addresses_strides_layouts.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/07_addressing_strides_and_layout.md`](../reference/07_addressing_strides_and_layout.md)

## Navigation

前: [`docs/modules/03_memory_hierarchy_and_device_query.md`](03_memory_hierarchy_and_device_query.md) / 次: [`docs/modules/05_hbm_bandwidth_and_copy.md`](05_hbm_bandwidth_and_copy.md)

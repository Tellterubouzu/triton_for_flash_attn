# Module 05: HBM copy bandwidth と effective bandwidth

## 位置づけ

単純 copy kernel の read/write bytes から effective bandwidth を計算し、memory-bound kernel の測定基準を作る。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| effective bandwidth | 実測時間と論理的な転送 byte 数から計算した帯域。理論帯域とは異なる。 |
| read traffic | HBM から読み出す byte 数。 |
| write traffic | HBM へ書き込む byte 数。 |
| memory-bound | 実行時間が演算器ではなく memory bandwidth に主に制限される状態。 |
| copy kernel | 入力を読み、出力へ書く最小の memory transfer kernel。 |

## 数式

この数式は、copy kernel の実測時間から effective bandwidth を計算する式です。read と write の両方を HBM traffic として数えます。

\[
\mathrm{bandwidth} = \frac{\mathrm{read\ bytes} + \mathrm{write\ bytes}}{\mathrm{elapsed\ seconds}}
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/05_hbm_bandwidth_and_copy.md`](../../exercises/05_hbm_bandwidth_and_copy.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/05_hbm_bandwidth_and_copy.py`](../../lessons/05_hbm_bandwidth_and_copy.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/01_performance_model.md`](../reference/01_performance_model.md)
- [`docs/reference/06_gpu_memory_hierarchy_low_level.md`](../reference/06_gpu_memory_hierarchy_low_level.md)

## Navigation

前: [`docs/modules/04_tensor_addresses_strides_layouts.md`](04_tensor_addresses_strides_layouts.md) / 次: [`docs/modules/06_coalescing_strides_cache.md`](06_coalescing_strides_cache.md)

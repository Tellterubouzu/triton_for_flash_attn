# Module 08: elementwise fusion と intermediate tensor 削減

## 位置づけ

PyTorch で複数 operator になる処理を 1 Triton kernel に融合し、HBM traffic と launch overhead の削減を観察する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| fusion | 複数の演算を 1 kernel にまとめ、中間 tensor の HBM read/write を消す最適化。 |
| intermediate tensor | 演算の途中結果として materialize される tensor。 |
| arithmetic intensity | 転送 byte あたりの FLOPs。低いほど memory-bound になりやすい。 |
| launch overhead | kernel 実行を開始する固定コスト。小さい演算では支配的になりやすい。 |

## 数式

この数式は、転送 byte あたりに何回の浮動小数点演算を行うかを表す arithmetic intensity です。値が低い演算は memory-bound になりやすいです。

\[
\mathrm{arithmetic\ intensity}=\frac{\mathrm{FLOPs}}{\mathrm{bytes\ moved}}
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/08_elementwise_fusion.md`](../../exercises/08_elementwise_fusion.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/08_elementwise_fusion.py`](../../lessons/08_elementwise_fusion.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/01_performance_model.md`](../reference/01_performance_model.md)

## Navigation

前: [`docs/modules/07_vector_add_first_kernel.md`](07_vector_add_first_kernel.md) / 次: [`docs/modules/09_cache_hints_and_eviction.md`](09_cache_hints_and_eviction.md)

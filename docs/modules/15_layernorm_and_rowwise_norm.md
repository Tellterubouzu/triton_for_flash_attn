# Module 15: LayerNorm と row-wise normalization

## 位置づけ

row-wise mean/variance reduction と affine 変換を 1 kernel にまとめる。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| LayerNorm | サンプル内の特徴次元で平均と分散を計算し正規化する層。 |
| mean | 平均。LayerNorm では行内特徴量の中心を表す。 |
| variance | 分散。特徴量のばらつきを表す。 |
| epsilon | 分母が 0 に近づくことを避けるために分散へ足す小さい値。 |
| affine | 正規化後に scale と bias を適用する線形変換。 |
| saved stats | backward で再利用する mean/rstd などの統計量。 |

## 数式

この数式は、1 行内の特徴量を平均 0・分散 1 に正規化し、scale と bias を適用する LayerNorm を表します。

\[
y_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\,w_i+b_i
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/15_layernorm_and_rowwise_norm.md`](../../exercises/15_layernorm_and_rowwise_norm.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/15_layernorm_and_rowwise_norm.py`](../../lessons/15_layernorm_and_rowwise_norm.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/16_numerics_correctness.md`](../reference/16_numerics_correctness.md)

## Navigation

前: [`docs/modules/14_tensor_cores_and_tl_dot.md`](14_tensor_cores_and_tl_dot.md) / 次: [`docs/modules/16_bottleneck_lab_and_profiler.md`](16_bottleneck_lab_and_profiler.md)

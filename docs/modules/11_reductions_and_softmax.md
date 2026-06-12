# Module 11: row-wise reduction と stable softmax

## 位置づけ

行ごとの max/sum reduction と数値安定化 softmax を Triton で書く。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| reduction | 複数要素から max/sum など 1 つまたは少数の値を作る演算。 |
| row-wise | 行ごとに独立して計算すること。softmax や LayerNorm で頻出。 |
| stable softmax | overflow を避けるため各行の最大値を引いてから exp を取る softmax。 |
| max trick | softmax の不変性を使い、logit から最大値を引く数値安定化。 |
| power-of-two block | Triton の reduction を扱いやすくするため、列数を 2 の冪へ丸めた block size。 |

## 数式

この数式は、各行の logits を確率分布へ変換する stable softmax です。行最大値を引くことで exp の overflow を避けます。

\[
y_{ij}=\frac{\exp(x_{ij}-m_i)}{\sum_k \exp(x_{ik}-m_i)},\quad m_i=\max_j x_{ij}
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/11_reductions_and_softmax.md`](../../exercises/11_reductions_and_softmax.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/11_reductions_and_softmax.py`](../../lessons/11_reductions_and_softmax.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/16_numerics_correctness.md`](../reference/16_numerics_correctness.md)

## Navigation

前: [`docs/modules/10_block_pointers_boundary_check.md`](10_block_pointers_boundary_check.md) / 次: [`docs/modules/12_numerics_and_correctness.md`](12_numerics_and_correctness.md)

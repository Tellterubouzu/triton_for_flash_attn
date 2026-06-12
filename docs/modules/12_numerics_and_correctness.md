# Module 12: numerics、dtype、correctness tolerance

## 位置づけ

bitwise 一致ではなく、dtype と演算順序に応じた誤差許容で kernel を検証する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| fp32 | 32-bit floating point。accumulator や reference 計算で使う標準的な精度。 |
| fp16 | 16-bit floating point。高速だが表現範囲と精度が狭い。 |
| bf16 | bfloat16。fp16 より mantissa は粗いが exponent 範囲が fp32 に近い。 |
| rounding | 実数を有限精度浮動小数へ丸めること。 |
| absolute error | 予測値と基準値の絶対差。 |
| relative error | 基準値の大きさに対する誤差。 |
| NaN/Inf | 未定義値や無限大。kernel 検証で必ず検出するべき異常値。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/12_numerics_and_correctness.md`](../../exercises/12_numerics_and_correctness.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/12_numerics_and_correctness.py`](../../lessons/12_numerics_and_correctness.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/16_numerics_correctness.md`](../reference/16_numerics_correctness.md)

## Navigation

前: [`docs/modules/11_reductions_and_softmax.md`](11_reductions_and_softmax.md) / 次: [`docs/modules/13_tiled_matmul.md`](13_tiled_matmul.md)

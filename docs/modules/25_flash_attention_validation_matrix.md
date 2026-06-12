# Module 25: validation matrix と correctness sweep

## 位置づけ

単一 shape だけでなく dtype/seq/head_dim/causal を sweep し、performance tuning 前に correctness を固定する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| validation matrix | dtype、shape、mask 条件などを組み合わせた検証ケース一覧。 |
| pass/fail/skip | 検証結果の状態。unsupported case を fail ではなく skip として管理する。 |
| shape sweep | seq/head_dim/batch/head 数などを変えながら検証すること。 |
| dtype sweep | fp32/fp16/bf16 など dtype を変えながら検証すること。 |
| determinism | 同じ入力で同じ出力が再現される性質。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/25_flash_attention_validation_matrix.md`](../../exercises/25_flash_attention_validation_matrix.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/25_flash_attention_validation_matrix.py`](../../lessons/25_flash_attention_validation_matrix.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/18_flash_attention_validation_and_portability.md`](../reference/18_flash_attention_validation_and_portability.md)

## Navigation

前: [`docs/modules/24_flash_attention_autotune_portability.md`](24_flash_attention_autotune_portability.md) / 次: [`docs/modules/26_flash_attention_backward_plan.md`](26_flash_attention_backward_plan.md)

# Module 26: FlashAttention backward design

## 位置づけ

forward kernel から backward kernel へ拡張する際に必要な gradient、保存値、再計算方針を整理する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| gradient | loss に対する tensor の微分。 |
| dQ/dK/dV | attention backward で求める Query/Key/Value の gradient。 |
| recomputation | memory 節約のため forward 中間値を保存せず backward で再計算する方針。 |
| saved tensor | backward のため forward で保存する tensor。 |
| dropout mask | dropout で残した要素を再現するための mask。 |
| causal gradient | causal mask を考慮した backward の gradient。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/26_flash_attention_backward_plan.md`](../../exercises/26_flash_attention_backward_plan.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/26_flash_attention_backward_plan.py`](../../lessons/26_flash_attention_backward_plan.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/19_flash_attention_variants_and_production.md`](../reference/19_flash_attention_variants_and_production.md)

## Navigation

前: [`docs/modules/25_flash_attention_validation_matrix.md`](25_flash_attention_validation_matrix.md) / 次: [`docs/modules/27_ncu_rocprof_workflow.md`](27_ncu_rocprof_workflow.md)

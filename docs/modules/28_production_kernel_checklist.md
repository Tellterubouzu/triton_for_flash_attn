# Module 28: production FlashAttention kernel checklist

## 位置づけ

研究用 kernel と production kernel の差分を整理し、CI、fallback、API、variant 対応を計画する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| API contract | kernel が受け付ける dtype、shape、layout、mask、error 処理の仕様。 |
| CI matrix | GPU/backend/PyTorch/Triton/dtype/shape の組み合わせを自動テストする表。 |
| fallback path | unsupported case で安全な PyTorch/SDPA 実装へ戻る経路。 |
| variable length | batch 内で sequence length が異なるケース。 |
| paged KV cache | LLM 推論で KV cache を page 単位で管理する方式。 |
| GQA/MQA | Grouped Query Attention / Multi-Query Attention。Q head と KV head の数が異なる attention variant。 |
| version pinning | compiler/runtime の version を固定し、再現性を確保すること。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/28_production_kernel_checklist.md`](../../exercises/28_production_kernel_checklist.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/28_production_kernel_checklist.py`](../../lessons/28_production_kernel_checklist.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/20_production_flash_attention_checklist.md`](../reference/20_production_flash_attention_checklist.md)
- [`docs/reference/19_flash_attention_variants_and_production.md`](../reference/19_flash_attention_variants_and_production.md)

## Navigation

前: [`docs/modules/27_ncu_rocprof_workflow.md`](27_ncu_rocprof_workflow.md)

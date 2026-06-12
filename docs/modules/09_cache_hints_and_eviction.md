# Module 09: cache modifier と eviction hint

## 位置づけ

`tl.load`/`tl.store` の cache hint を性能実験の対象として扱い、保証ではなく hint であることを理解する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| cache modifier | load/store が cache をどう使うかを compiler/backend に伝える hint。 |
| eviction policy | cache に長く残したいか早く追い出してよいかを伝える hint。 |
| .ca | NVIDIA PTX の load cache modifier 例。通常の cache-all 的な意味で使われる。 |
| .cg | global/L2 寄りの cache 方針を示す modifier。 |
| .cv | volatile に近い再読込を促す modifier。 |
| hint | 性能上の希望を伝える指定。意味や効果は backend/GPU/compiler に依存する。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/09_cache_hints_and_eviction.md`](../../exercises/09_cache_hints_and_eviction.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/09_cache_hints_and_eviction.py`](../../lessons/09_cache_hints_and_eviction.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/10_cache_hints_ir_and_profiling.md`](../reference/10_cache_hints_ir_and_profiling.md)

## Navigation

前: [`docs/modules/08_elementwise_fusion.md`](08_elementwise_fusion.md) / 次: [`docs/modules/10_block_pointers_boundary_check.md`](10_block_pointers_boundary_check.md)

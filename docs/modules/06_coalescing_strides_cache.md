# Module 06: coalescing、stride、cache locality

## 位置づけ

連続 access と stride access の違いを測り、warp/program が発行する memory transaction の効率を理解する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| coalescing | 近い address への複数 access を少数の memory transaction にまとめること。 |
| memory transaction | GPU memory subsystem が実際に処理する転送単位。 |
| cache line/sector | cache がまとめて保持・転送する memory 範囲。 |
| stride access | 隣接要素ではなく一定間隔で memory を読む access pattern。 |
| cache hit | 必要な data が cache に存在し、HBM まで行かずに読めること。 |
| locality | 近い時間または近い address の data を再利用できる性質。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/06_coalescing_strides_cache.md`](../../exercises/06_coalescing_strides_cache.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/06_coalescing_strides_cache.py`](../../lessons/06_coalescing_strides_cache.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/10_cache_hints_ir_and_profiling.md`](../reference/10_cache_hints_ir_and_profiling.md)

## Navigation

前: [`docs/modules/05_hbm_bandwidth_and_copy.md`](05_hbm_bandwidth_and_copy.md) / 次: [`docs/modules/07_vector_add_first_kernel.md`](07_vector_add_first_kernel.md)

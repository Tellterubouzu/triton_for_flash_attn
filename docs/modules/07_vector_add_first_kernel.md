# Module 07: 最初の Triton kernel: vector add

## 位置づけ

`tl.program_id`, `tl.arange`, `tl.load`, `tl.store`, mask を使って最小 kernel を書く。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| tl.program_id | 現在の Triton program instance の index を返す関数。 |
| tl.arange | tile 内の vector offset を作る Triton primitive。 |
| tl.load | pointer から data を読み込む操作。mask と other で範囲外を安全に処理できる。 |
| tl.store | pointer へ data を書き込む操作。mask で範囲外 write を防ぐ。 |
| mask | 有効な lane/要素だけを load/store/計算するための boolean tensor。 |
| BLOCK_SIZE | 1 program が処理する要素数を表す compile-time constant。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/07_vector_add_first_kernel.md`](../../exercises/07_vector_add_first_kernel.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/07_vector_add_first_kernel.py`](../../lessons/07_vector_add_first_kernel.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/02_triton_programming_model.md`](../reference/02_triton_programming_model.md)

## Navigation

前: [`docs/modules/06_coalescing_strides_cache.md`](06_coalescing_strides_cache.md) / 次: [`docs/modules/08_elementwise_fusion.md`](08_elementwise_fusion.md)

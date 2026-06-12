# Module 10: block pointer、boundary_check、tl.advance

## 位置づけ

2D tile の境界処理を手書き mask だけでなく block pointer で表現する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| block pointer | base pointer、shape、stride、offset、block shape をまとめた Triton の tile pointer。 |
| tl.make_block_ptr | block pointer を作る Triton API。 |
| boundary_check | block pointer load/store で境界外次元を自動的に処理する指定。 |
| padding_option | 境界外 load に zero などを入れる指定。 |
| tl.advance | block pointer の offsets を進めて次 tile を指す操作。 |
| tensor descriptor | tensor の形状・stride・layout を kernel に伝える情報のまとまり。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/10_block_pointers_boundary_check.md`](../../exercises/10_block_pointers_boundary_check.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/10_block_pointers_boundary_check.py`](../../lessons/10_block_pointers_boundary_check.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/12_block_pointers_tensor_descriptors.md`](../reference/12_block_pointers_tensor_descriptors.md)

## Navigation

前: [`docs/modules/09_cache_hints_and_eviction.md`](09_cache_hints_and_eviction.md) / 次: [`docs/modules/11_reductions_and_softmax.md`](11_reductions_and_softmax.md)

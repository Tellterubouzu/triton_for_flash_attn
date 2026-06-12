# Module 19: streams、transfers、overlap

## 位置づけ

host-device copy、kernel 実行、非同期 stream の関係を理解し、benchmark の同期点を誤らない。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| stream | GPU 上の操作列。異なる stream は条件が揃えば並行実行できる。 |
| default stream | 明示指定しない GPU 操作が入る既定の stream。 |
| pinned memory | page lock された host memory。GPU との非同期転送に有利。 |
| async copy | CPU が完了を待たずに enqueue する転送。 |
| overlap | copy と compute などを時間的に重ねること。 |
| event | GPU stream 上の時刻・同期点を表す object。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/19_streams_transfers_overlap.md`](../../exercises/19_streams_transfers_overlap.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/19_streams_transfers_overlap.py`](../../lessons/19_streams_transfers_overlap.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/09_memory_transfer_and_allocator.md`](../reference/09_memory_transfer_and_allocator.md)

## Navigation

前: [`docs/modules/18_ir_ptx_sass_inspection.md`](18_ir_ptx_sass_inspection.md) / 次: [`docs/modules/20_allocator_zero_fill_lifetime.md`](20_allocator_zero_fill_lifetime.md)

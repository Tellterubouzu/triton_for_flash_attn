# Module 20: allocator、zero fill、tensor lifetime

## 位置づけ

allocation、cache、zero fill、実際の HBM write を分離して観察する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| caching allocator | PyTorch が GPU memory を再利用するために保持する allocator。 |
| allocated memory | Tensor が現在使用している memory 量。 |
| reserved memory | allocator が再利用のために確保したまま保持している memory 量。 |
| zero fill | memory 領域を 0 で初期化する write 操作。 |
| empty_cache | 未使用 cache block を解放し、他 process から見える空き容量を増やす API。 |
| fragmentation | 空き memory が細切れになり、大きな連続領域を取りにくくなる状態。 |
| tensor lifetime | Tensor が作られてから参照されなくなるまでの期間。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/20_allocator_zero_fill_lifetime.md`](../../exercises/20_allocator_zero_fill_lifetime.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/20_allocator_zero_fill_lifetime.py`](../../lessons/20_allocator_zero_fill_lifetime.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/09_memory_transfer_and_allocator.md`](../reference/09_memory_transfer_and_allocator.md)

## Navigation

前: [`docs/modules/19_streams_transfers_overlap.md`](19_streams_transfers_overlap.md) / 次: [`docs/modules/21_attention_pytorch_baselines.md`](21_attention_pytorch_baselines.md)

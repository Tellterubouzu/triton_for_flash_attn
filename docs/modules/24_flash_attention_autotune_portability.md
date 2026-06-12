# Module 24: autotune と multi-GPU portability

## 位置づけ

GPU、dtype、shape ごとに最適 config が変わることを前提に、portable default と tuned config を分ける。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| autotune | 複数 config を実測し、対象 shape/GPU で速いものを選ぶ仕組み。 |
| config | BLOCK_M/N、num_warps、num_stages など kernel の compile-time parameter の組。 |
| meta-parameter | Triton kernel compile 時に固定される parameter。 |
| portability | 複数 GPU/backend で正しく動き、妥当な性能を出せる性質。 |
| architecture-specific tuning | 特定 GPU 世代・backend に合わせた tuning。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/24_flash_attention_autotune_portability.md`](../../exercises/24_flash_attention_autotune_portability.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/24_flash_attention_autotune_portability.py`](../../lessons/24_flash_attention_autotune_portability.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/05_multi_gpu_portability.md`](../reference/05_multi_gpu_portability.md)
- [`docs/reference/18_flash_attention_validation_and_portability.md`](../reference/18_flash_attention_validation_and_portability.md)

## Navigation

前: [`docs/modules/23_flash_attention_forward_triton.md`](23_flash_attention_forward_triton.md) / 次: [`docs/modules/25_flash_attention_validation_matrix.md`](25_flash_attention_validation_matrix.md)

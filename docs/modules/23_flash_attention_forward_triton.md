# Module 23: FlashAttention forward Triton kernel

## 位置づけ

Query block と Key/Value block を走査し、online softmax で forward 出力を計算する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| BLOCK_M | 1 program が担当する query 行数。 |
| BLOCK_N | 1 回に読む key/value 列数。 |
| BLOCK_D | head dimension 方向の block size。 |
| causal boundary block | causal mask で full skip でも full valid でもなく、一部だけ有効な block。 |
| Q/K/V tile | HBM から読み込む Query/Key/Value の小ブロック。 |
| normalization | 最後に accumulator を softmax 分母で割って output にする処理。 |

## 数式

この数式は、online softmax の accumulator を softmax 分母で割って最終出力に変換する正規化です。FlashAttention forward の最後に行います。

\[
O_i=\frac{acc_i}{l_i}
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/23_flash_attention_forward_triton.md`](../../exercises/23_flash_attention_forward_triton.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/23_flash_attention_forward_triton.py`](../../lessons/23_flash_attention_forward_triton.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/11_flash_attention_io_schedule.md`](../reference/11_flash_attention_io_schedule.md)
- [`docs/reference/03_flash_attention_math.md`](../reference/03_flash_attention_math.md)

## Navigation

前: [`docs/modules/22_flash_attention_io_accounting.md`](22_flash_attention_io_accounting.md) / 次: [`docs/modules/24_flash_attention_autotune_portability.md`](24_flash_attention_autotune_portability.md)

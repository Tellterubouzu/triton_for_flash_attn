# Module 22: FlashAttention IO accounting と online softmax

## 位置づけ

naive attention と FlashAttention の HBM temporary/traffic を比較し、online softmax の状態変数を理解する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| IO complexity | 演算ではなく memory read/write の量を中心に見た計算量。 |
| online softmax | 全列を一度に保持せず、block ごとに max と分母を更新する softmax。 |
| m statistic | softmax の数値安定化に使う行ごとの現在最大値。 |
| l statistic | 再スケール済み softmax 分母。 |
| output accumulator | 正規化前の weighted value sum を保持する一時値。 |
| exact attention | 近似ではなく naive softmax attention と同じ数学的結果を目指す attention。 |

## 数式

この数式は、FlashAttention の online softmax で、過去 block と新しい block の最大値を統合する更新です。続く 2 式は softmax 分母と出力 accumulator を再スケールしながら更新します。

\[
m_{new}=\max(m_{old},\max S_{block})
\]
\[
l_{new}=e^{m_{old}-m_{new}}l_{old}+\sum_j e^{S_j-m_{new}}
\]
\[
acc_{new}=e^{m_{old}-m_{new}}acc_{old}+\sum_j e^{S_j-m_{new}}V_j
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/22_flash_attention_io_accounting.md`](../../exercises/22_flash_attention_io_accounting.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/22_flash_attention_io_accounting.py`](../../lessons/22_flash_attention_io_accounting.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/11_flash_attention_io_schedule.md`](../reference/11_flash_attention_io_schedule.md)

## Navigation

前: [`docs/modules/21_attention_pytorch_baselines.md`](21_attention_pytorch_baselines.md) / 次: [`docs/modules/23_flash_attention_forward_triton.md`](23_flash_attention_forward_triton.md)

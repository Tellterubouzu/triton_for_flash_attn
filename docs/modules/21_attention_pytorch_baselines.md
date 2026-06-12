# Module 21: PyTorch attention baseline と materialization

## 位置づけ

naive attention がどの中間 tensor を HBM に作るかを確認し、FlashAttention が解く問題を具体化する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| Query/Key/Value | attention の入力 tensor。Query は問い合わせ、Key は照合対象、Value は集約される情報を表す。 |
| logits | softmax 前の score。attention では QK^T / sqrt(d)。 |
| causal mask | 未来 token を参照しないように score を -inf にする mask。 |
| softmax probability | logits を正規化した attention weight。 |
| SDPA | scaled dot-product attention。PyTorch では最適化 backend を選ぶ API も含む。 |
| materialization | 中間結果を実際の tensor として memory に書き出すこと。 |

## 数式

この数式は、Query と Key の類似度を softmax で正規化し、Value の加重和を取る scaled dot-product attention を表します。

\[
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/21_attention_pytorch_baselines.md`](../../exercises/21_attention_pytorch_baselines.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/21_attention_pytorch_baselines.py`](../../lessons/21_attention_pytorch_baselines.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/03_flash_attention_math.md`](../reference/03_flash_attention_math.md)

## Navigation

前: [`docs/modules/20_allocator_zero_fill_lifetime.md`](20_allocator_zero_fill_lifetime.md) / 次: [`docs/modules/22_flash_attention_io_accounting.md`](22_flash_attention_io_accounting.md)

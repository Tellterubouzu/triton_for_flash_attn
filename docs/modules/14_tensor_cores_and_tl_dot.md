# Module 14: Tensor Core、MFMA、MMA、tl.dot lowering

## 位置づけ

`tl.dot` が backend に応じて Tensor Core/MFMA 系命令へ lower されるかを生成コードと profiler で確認する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| Tensor Core | NVIDIA GPU の行列積専用演算器。fp16/bf16/tf32 等の matrix multiply-accumulate を高スループットで実行する。 |
| MMA | Matrix Multiply-Accumulate。小さな行列ブロックの積和命令。 |
| MFMA | AMD GPU の matrix fused multiply-add 系命令。 |
| tl.dot | Triton の block 行列積 primitive。FlashAttention の QK^T と PV で中核になる。 |
| TF32 | NVIDIA Ampere 以降で fp32 入力を Tensor Core 向けに扱う形式。 |
| input_precision | `tl.dot` で fp32 入力の精度方針などを指定する引数。 |
| lowering | Triton の高水準表現を LLVM/PTX/ISA など低水準表現へ変換すること。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/14_tensor_cores_and_tl_dot.md`](../../exercises/14_tensor_cores_and_tl_dot.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/14_tensor_cores_and_tl_dot.py`](../../lessons/14_tensor_cores_and_tl_dot.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/14_tensor_cores_mma_and_tl_dot.md`](../reference/14_tensor_cores_mma_and_tl_dot.md)

## Navigation

前: [`docs/modules/13_tiled_matmul.md`](13_tiled_matmul.md) / 次: [`docs/modules/15_layernorm_and_rowwise_norm.md`](15_layernorm_and_rowwise_norm.md)

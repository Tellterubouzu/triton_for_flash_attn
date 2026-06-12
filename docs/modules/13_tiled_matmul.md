# Module 13: tiled matmul と SRAM reuse

## 位置づけ

GEMM を tile に分割し、K loop と accumulator によって HBM read を再利用する基本形を理解する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| GEMM | General Matrix-Matrix Multiplication。C = A @ B の一般的な行列積。 |
| tile | 行列の小さな矩形 block。1 program が 1 output tile を担当する。 |
| K loop | matmul の内積次元 K を BLOCK_K ごとに走査する loop。 |
| accumulator | 部分和を保持する一時領域。通常 fp32 で持つ。 |
| SRAM reuse | on-chip 側に読んだ tile を複数演算で使い回し、HBM access を減らすこと。 |
| grouped ordering | program の実行順を並べ替え、L2 reuse を改善する matmul の scheduling 技法。 |

## 数式

この数式は、行列 A と B の積 C を定義する GEMM です。Triton では C の tile を program に割り当て、K 方向を block ごとに畳み込みます。

\[
C_{ij}=\sum_{k=0}^{K-1} A_{ik}B_{kj}
\]

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/13_tiled_matmul.md`](../../exercises/13_tiled_matmul.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/13_tiled_matmul.py`](../../lessons/13_tiled_matmul.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/01_performance_model.md`](../reference/01_performance_model.md)

## Navigation

前: [`docs/modules/12_numerics_and_correctness.md`](12_numerics_and_correctness.md) / 次: [`docs/modules/14_tensor_cores_and_tl_dot.md`](14_tensor_cores_and_tl_dot.md)

# Module 18: IR、PTX、SASS inspection

## 位置づけ

Triton が生成する中間表現・アセンブリを保存し、性能問題の仮説を立てる入口を作る。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| TTIR | Triton Tensor IR。Triton の高水準中間表現。 |
| TTGIR | Triton GPU IR。GPU 実行に近づいた中間表現。 |
| LLVM IR | LLVM compiler infrastructure の中間表現。 |
| PTX | NVIDIA GPU 向けの仮想 ISA。 |
| SASS | NVIDIA GPU の実機械語に近い assembly。 |
| register spill | register に収まらない値が local memory へ退避されること。 |
| assembly token | `mma` や `ldmatrix` など生成コード上の命令断片。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/18_ir_ptx_sass_inspection.md`](../../exercises/18_ir_ptx_sass_inspection.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/18_ir_ptx_sass_inspection.py`](../../lessons/18_ir_ptx_sass_inspection.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/10_cache_hints_ir_and_profiling.md`](../reference/10_cache_hints_ir_and_profiling.md)

## Navigation

前: [`docs/modules/17_torch_compile_and_custom_triton.md`](17_torch_compile_and_custom_triton.md) / 次: [`docs/modules/19_streams_transfers_overlap.md`](19_streams_transfers_overlap.md)

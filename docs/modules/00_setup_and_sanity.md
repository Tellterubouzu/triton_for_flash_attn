# Module 00: セットアップ、同期、benchmark の前提

## 位置づけ

GPU/Triton が使えるか、測定時に何を同期する必要があるか、warmup と繰り返し計測がなぜ必要かを確認する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| GPU | 多数の演算器を持つ並列計算装置。Triton kernel は主に GPU 上で実行される。 |
| CUDA | NVIDIA GPU 向けのプログラミング環境と runtime。 |
| ROCm | AMD GPU 向けの計算ソフトウェアスタック。 |
| Triton | Python から GPU kernel を記述し、GPU 向けコードへコンパイルする言語・コンパイラ。 |
| kernel | GPU 上で多数の並列実行単位として走る関数。CPU の通常関数呼び出しとは実行モデルが異なる。 |
| synchronization | CPU が GPU の非同期処理の完了を待つこと。benchmark では必須。 |
| warmup | 初回コンパイル、allocator 初期化、cache 状態などを本計測から除くための予備実行。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/00_setup_and_sanity.md`](../../exercises/00_setup_and_sanity.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/00_setup_and_sanity.py`](../../lessons/00_setup_and_sanity.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## Navigation

次: [`docs/modules/01_gpu_execution_model.md`](01_gpu_execution_model.md)

# Module 16: PyTorch profiler で bottleneck を選ぶ

## 位置づけ

PyTorch 実装を見て、どの operator/kernel を Triton 化する価値があるかを判断する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| profiler | 実行時間、memory、呼び出し回数などを記録する測定ツール。 |
| hotspot | 全体時間への寄与が大きい箇所。 |
| operator | PyTorch の `aten::matmul` などフレームワーク上の演算単位。 |
| CUDA/HIP kernel time | 実際に GPU 上で動いた kernel の実行時間。 |
| trace | 時間軸上の CPU/GPU activity を記録したデータ。 |
| self time | 子呼び出しを除いたその operator 自身の時間。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/16_bottleneck_lab_and_profiler.md`](../../exercises/16_bottleneck_lab_and_profiler.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/16_bottleneck_lab_and_profiler.py`](../../lessons/16_bottleneck_lab_and_profiler.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/04_bottleneck_workflow.md`](../reference/04_bottleneck_workflow.md)
- [`docs/reference/17_profiler_metrics_ncu_rocprof.md`](../reference/17_profiler_metrics_ncu_rocprof.md)

## Navigation

前: [`docs/modules/15_layernorm_and_rowwise_norm.md`](15_layernorm_and_rowwise_norm.md) / 次: [`docs/modules/17_torch_compile_and_custom_triton.md`](17_torch_compile_and_custom_triton.md)

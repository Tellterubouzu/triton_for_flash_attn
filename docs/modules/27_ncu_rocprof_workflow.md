# Module 27: Nsight Compute / Nsight Systems / rocprof workflow

## 位置づけ

PyTorch profiler 後に外部 profiler でどの metric を見るか、NVIDIA/AMD で手順を分けて整理する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| Nsight Compute | NVIDIA CUDA kernel の詳細 metric を見る profiler。 |
| Nsight Systems | CPU/GPU/stream の時間軸を俯瞰する profiler。 |
| rocprof | AMD ROCm 環境で GPU kernel metric を収集する profiler。 |
| metric | bandwidth、occupancy、stall など profiler が計測する指標。 |
| tensor pipe utilization | Tensor Core/MFMA 系演算器の使用率に近い metric。 |
| stall reason | warp が実行できず待っている理由。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/27_ncu_rocprof_workflow.md`](../../exercises/27_ncu_rocprof_workflow.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/27_ncu_rocprof_workflow.py`](../../lessons/27_ncu_rocprof_workflow.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/17_profiler_metrics_ncu_rocprof.md`](../reference/17_profiler_metrics_ncu_rocprof.md)

## Navigation

前: [`docs/modules/26_flash_attention_backward_plan.md`](26_flash_attention_backward_plan.md) / 次: [`docs/modules/28_production_kernel_checklist.md`](28_production_kernel_checklist.md)

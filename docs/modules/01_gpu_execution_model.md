# Module 01: kernel、launch、grid、Triton program、warp

## 位置づけ

CUDA の grid/block/thread と Triton の program/tile の対応を理解し、FlashAttention の query block がどの program に割り当たるかを説明できるようにする。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| launch | CPU 側から GPU kernel の実行を開始する操作。launch 自体にも固定 overhead がある。 |
| grid | kernel launch で作られる実行単位全体の集合。Triton では grid が program instance 数を決める。 |
| thread block | CUDA で 1 つの SM に配置される thread のまとまり。Triton では直接 threadIdx を書かず program 単位で考える。 |
| Triton program instance | Triton kernel の 1 つの実行単位。`tl.program_id` でどの instance かを取得する。 |
| tile | 1 program がまとめて処理する連続または矩形のデータブロック。 |
| warp | NVIDIA GPU で 32 threads が SIMT として一緒に命令を実行する単位。 |
| SIMT | Single Instruction, Multiple Threads。複数 thread が同じ命令列を異なるデータに適用する実行方式。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/01_gpu_execution_model.md`](../../exercises/01_gpu_execution_model.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/01_gpu_execution_model.py`](../../lessons/01_gpu_execution_model.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/00_gpu_execution_model.md`](../reference/00_gpu_execution_model.md)

## Navigation

前: [`docs/modules/00_setup_and_sanity.md`](00_setup_and_sanity.md) / 次: [`docs/modules/02_warps_sms_occupancy.md`](02_warps_sms_occupancy.md)

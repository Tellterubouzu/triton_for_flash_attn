# Module 02: SM、warp、occupancy、num_warps

## 位置づけ

`num_warps` が 1 program の実行資源と並列性にどう影響するかを粗く見積もる。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| SM | Streaming Multiprocessor。GPU の主要な実行ユニットで、warp scheduler、register file、shared memory などを持つ。 |
| occupancy | SM に同時滞在できる warp/thread/program の割合。高ければ常に速いわけではない。 |
| resident program | ある時点で SM 上に配置されている program instance。 |
| warp scheduler | 実行可能な warp を選んで命令を発行するハードウェア。 |
| register pressure | kernel が必要とする register 数の多さ。増えすぎると occupancy 低下や spill につながる。 |
| num_warps | Triton の meta-parameter。1 program の中で使う warp 数の指定。 |
| num_stages | Triton の pipelining 段数を指定する meta-parameter。主に load と compute の重なりに関係する。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/02_warps_sms_occupancy.md`](../../exercises/02_warps_sms_occupancy.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/02_warps_sms_occupancy.py`](../../lessons/02_warps_sms_occupancy.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/13_kernel_warp_occupancy.md`](../reference/13_kernel_warp_occupancy.md)

## Navigation

前: [`docs/modules/01_gpu_execution_model.md`](01_gpu_execution_model.md) / 次: [`docs/modules/03_memory_hierarchy_and_device_query.md`](03_memory_hierarchy_and_device_query.md)

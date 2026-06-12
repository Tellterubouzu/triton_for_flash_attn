# Module 03: HBM、L2、shared memory、register と device query

## 位置づけ

GPU のメモリ階層と、Python/PyTorch から取得できる容量・属性、取得できない詳細を区別する。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| HBM | High Bandwidth Memory。GPU board 上の大容量 global memory。FlashAttention では HBM read/write 削減が主目標になる。 |
| L2 cache | SM 群で共有される cache。HBM より速く小さい。 |
| SRAM | GPU chip 上の高速小容量メモリの総称。shared memory や register file を説明するときに使う。 |
| shared memory | SM 内で thread block/program が使う高速な on-chip memory。Triton では明示配列ではなく tile と compiler 管理として現れることが多い。 |
| register file | 各 thread/program の一時値を保持する最も近い記憶資源。 |
| bandwidth | 単位時間あたりに転送できる byte 数。 |
| latency | 1 回の memory access の開始から完了までの遅延。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/03_memory_hierarchy_and_device_query.md`](../../exercises/03_memory_hierarchy_and_device_query.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/03_memory_hierarchy_and_device_query.py`](../../lessons/03_memory_hierarchy_and_device_query.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/06_gpu_memory_hierarchy_low_level.md`](../reference/06_gpu_memory_hierarchy_low_level.md)
- [`docs/reference/08_hbm_l2_shared_registers.md`](../reference/08_hbm_l2_shared_registers.md)

## Navigation

前: [`docs/modules/02_warps_sms_occupancy.md`](02_warps_sms_occupancy.md) / 次: [`docs/modules/04_tensor_addresses_strides_layouts.md`](04_tensor_addresses_strides_layouts.md)

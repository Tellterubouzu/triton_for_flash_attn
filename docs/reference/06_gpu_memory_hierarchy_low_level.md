# 06. GPU memory hierarchy from a Triton programmer's view

この章の目的は、Triton kernel を「数式の高速化」ではなく「どのメモリ階層から何 byte 読み、どこに何 byte 書くか」として読むことです。

## 用語

この教材では次の名前を使います。

- **HBM / global memory**: GPU device memory。PyTorch Tensor の実体は通常ここにあります。容量は数十 GiB 程度ですが、on-chip memory より遅いです。
- **L2 cache**: GPU 全体または GPC/cluster レベルで共有される cache。HBM への再アクセスを減らします。
- **L1 / shared memory / SRAM**: SM に近い on-chip memory。CUDA C++ では shared memory を明示的に宣言できます。Triton では通常、`tl.load` した block tensor、`tl.dot` の tile、accumulator が compiler により register/shared/L1 側へ配置されます。
- **register**: thread/warp が保持する最も近い storage。Triton の scalar や block tensor の一部は register に置かれます。増えすぎると occupancy が落ちたり spill が起きます。

重要なのは、Triton では CUDA C++ の `extern __shared__` のように shared memory buffer を直接宣言するのではなく、tile shape、`tl.load`、`tl.dot`、`num_warps`、`num_stages`、cache hint を通じて compiler に配置を決めさせることです。

## 容量の取得

HBM の総量・空き容量は `torch.cuda.mem_get_info()` から取得します。`torch.cuda.get_device_properties()` からは `total_memory`、`shared_memory_per_block`、`multi_processor_count`、`warp_size` など、実装判断に必要な属性を取得できます。ただし、属性名は CUDA/ROCm/PyTorch の組み合わせで差があるため、教材では `hardware.get_device_report()` で安全に introspection します。

```python
from triton_flash_course.hardware import print_device_report
print_device_report()
```

## HBM と SRAM の読み替え

FlashAttention の文脈で「SRAM に置く」という表現は、Triton のソース上では「`q` block, `k` block, `v` block, accumulator を block tensor として持ち、HBM に中間行列を書かない」という意味で使います。実際にそれが register、shared memory、L1 のどこに置かれるかは、compiler lowering と backend に依存します。

## 消去・解放・zero fill

- `torch.empty` はメモリ領域を確保しますが、内容を初期化しません。
- `torch.zeros` や `tensor.zero_()` は HBM に zero を書きます。
- Triton の fill kernel でも zero を書けます。
- `del tensor` は Python reference を消すだけです。PyTorch の caching allocator は再利用のために memory block を保持することがあります。
- `torch.cuda.empty_cache()` は未使用 cache を driver 側へ返すための操作で、tensor が使っている memory を消すものではありません。

セキュリティ上の「完全消去」は、この教材の範囲外です。GPU allocator、cache、driver、peer access、unified memory の影響があるため、単に zero kernel を一度走らせれば完全消去になるとは仮定しません。

## FlashAttention に接続する見方

naive attention はおおまかに次の HBM traffic を持ちます。

1. Q, K を読み、score matrix `S = QK^T` を HBM に書く。
2. `S` を読み、softmax probability `P` を HBM に書く。
3. `P`, V を読み、O を HBM に書く。

FlashAttention は `S` と `P` を HBM に materialize しないように、K/V block を stream しながら online softmax の `m`, `l`, `acc` を更新します。これが低レイヤーで見たときの主な勝ち筋です。

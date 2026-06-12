# GPU 実行モデル入門: kernel, grid, block, program, warp

この章の目的は、Triton kernel を読むときに出てくる実行単位を、CUDA の低レイヤー概念と対応づけることです。FlashAttention の性能は、数式だけでは決まりません。どの単位で query block を割り当て、どの単位で key/value tile を読み、どの単位で SRAM/register に保持し、どれだけの warp を使って `tl.dot` と reduction を並列化するかで大きく変わります。

## kernel とは何か

kernel とは、GPU 上で実行される関数です。CPU 側の Python / C++ コードは host code、GPU 側で実行されるコードは device code と考えます。kernel launch とは、CPU が GPU に対して「この device code を、この入力、この grid 形状で実行せよ」と投入する操作です。

CUDA の表現では、kernel は多数の thread によって並列実行されます。

```cpp
my_kernel<<<grid_dim, block_dim>>>(args...);
```

Triton の表現では、`@triton.jit` を付けた Python 関数が kernel で、`kernel[grid](args...)` によって launch します。

```python
@triton.jit
def my_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x + 1.0, mask=mask)

my_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
```

重要なのは、Triton では CUDA の `threadIdx.x` のような「個々の thread id」を通常は直接書かないことです。代わりに `tl.program_id` で program instance の ID を得て、`tl.arange` で tile 内の offset vector を作ります。

## grid, block, program instance

CUDA では、kernel launch 時に大量の thread が作られ、それらは thread block にまとめられます。thread block は grid にまとめられます。1 つの thread block は 1 つの SM、Streaming Multiprocessor、上で実行されます。block 間の実行順序は保証されないため、同じ grid 内の別 block の結果に依存する kernel は基本的に書けません。

Triton では、CUDA の thread block に近い単位として program instance を考えます。たとえば vector add で `BLOCK=1024` なら、1 program instance が 1024 要素の tile を担当します。

```text
N = 5000, BLOCK = 1024
num_programs = ceil(5000 / 1024) = 5

program 0: offsets 0..1023
program 1: offsets 1024..2047
program 2: offsets 2048..3071
program 3: offsets 3072..4095
program 4: offsets 4096..5119, mask で 5000 以上を無効化
```

Triton の program は「実行単位」、tile は「データ単位」です。多くの単純 kernel では 1 program が 1 tile を処理しますが、FlashAttention では 1 program が 1 つの query block と head/batch の組を担当し、その中で key/value block を loop するような設計になります。

## warp とは何か

warp は、NVIDIA GPU の SIMT 実行における thread の束です。NVIDIA CUDA の warp size は 32 です。warp 内の thread は同じ命令列を進めますが、各 thread は異なる data lane を処理できます。

```text
1 warp = 32 lanes
lane 0, lane 1, ..., lane 31
```

warp 内で分岐条件が lane ごとに異なると、一部 lane を mask しながら分岐先を順に実行するため、実効利用率が下がります。これを warp divergence と呼びます。

```cpp
if (threadIdx.x % 2 == 0) {
    // 偶数 lane だけ実行される。奇数 lane は mask される。
}
```

FlashAttention では、warp divergence よりも、tile 形状、memory coalescing、`tl.dot` が専用行列演算へ落ちるか、register/shared memory 使用量で occupancy が下がりすぎないかがまず重要です。ただし causal mask の境界 block では条件分岐や mask が増えるため、branch と mask の扱いは無視できません。

## Triton の `num_warps`

Triton では kernel launch 時の meta-parameter として `num_warps` を指定できます。

```python
kernel[grid](..., BLOCK_M=64, BLOCK_N=64, num_warps=4)
```

これは「1 program instance の実行に使う warp 数」の指定です。NVIDIA GPU であれば、概念上は `num_warps * 32` lanes 相当の実行資源を 1 program に割り当てると考えると理解しやすいです。ただし、Triton は CUDA の per-thread code を直接書くモデルではないため、`num_warps=4` と書いても Python コード内に 128 個の明示 thread が見えるわけではありません。Triton compiler が block operation を thread/warp に分割します。

`num_warps` を増やすと、1 program 内の大きな tile や reduction や `tl.dot` を並列化しやすくなります。一方で、1 program が使う warp 数が増えるため、1 SM に同時常駐できる program 数は減りやすくなります。FlashAttention では、`BLOCK_M`, `BLOCK_N`, `BLOCK_D`, `num_warps`, `num_stages` をセットで sweep する必要があります。

## SIMT と Triton の block programming model

CUDA の古典的な書き方では、1 thread が 1 要素または複数要素を担当する per-thread code を書きます。Triton では、`tl.arange` で作った block tensor に対して vectorized に演算を書きます。

```python
# CUDA 的な発想
idx = blockIdx.x * blockDim.x + threadIdx.x
out[idx] = a[idx] + b[idx]

# Triton 的な発想
pid = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)
out = tl.load(a + offsets) + tl.load(b + offsets)
tl.store(c + offsets, out)
```

この差は FlashAttention で特に重要です。CUDA では warp-level primitive、shared memory、barrier、Tensor Core 命令の scheduling を明示的に設計することが多いです。Triton では、Q/K/V tile を block tensor として読み、`tl.dot`、`tl.max`、`tl.sum` を書き、compiler に thread/warp への分配や一部の shared memory 管理を任せます。

## kernel launch overhead

kernel launch は無料ではありません。小さい演算を PyTorch で多数並べると、各 op が別 kernel launch になり、計算そのものより launch overhead と HBM read/write が支配的になることがあります。

たとえば次の PyTorch 実装は読みやすいですが、複数の中間 tensor と複数の kernel を生みやすいです。

```python
y = torch.exp(x - x.max(dim=-1, keepdim=True).values)
y = y / y.sum(dim=-1, keepdim=True)
```

Triton では、1 program が 1 row を読み、max、exp、sum、divide、store を 1 kernel に融合できます。これが fused softmax の基本です。FlashAttention でも、`QK^T`、softmax、`PV` を別々の kernel と中間行列で処理するのではなく、1 つの kernel 内で block ごとに進めます。

## FlashAttention における対応

FlashAttention forward の典型的な Triton mapping は次のようになります。

```text
program_id(0): query block index
program_id(1): batch-head index

1 program:
  Q[batch, head, q_start:q_start+BLOCK_M, :]
  を担当する。

program 内 loop:
  K/V を BLOCK_N ごとに読む。
  S = Q K^T / sqrt(d) を tl.dot で計算する。
  online softmax の m, l, acc を更新する。
  最後に O block を HBM に store する。
```

このとき、`BLOCK_M` は 1 program が担当する query 数、`BLOCK_N` は一度に読む key/value 数、`BLOCK_D` は head dimension です。`num_warps` は program 内の dot/reduction をどれだけ並列化するかに影響します。

## CUDA と Triton の用語対応

| 概念 | CUDA | Triton | FlashAttention での意味 |
|---|---|---|---|
| GPU 上の関数 | kernel | `@triton.jit` function | attention forward kernel |
| 起動 | kernel launch | `kernel[grid](...)` | batch/head/query blocks を投入 |
| 大きな実行集合 | grid | grid | すべての query block と batch/head |
| 実行単位 | thread block / CTA | program instance | 1 個の query block |
| 実行単位 ID | `blockIdx` | `tl.program_id` | どの query block/head か |
| lane offset | `threadIdx` | `tl.arange` | tile 内の row/col offset |
| thread 束 | warp | `num_warps` で指定 | dot/reduction の並列度 |
| 行列専用演算 | Tensor Core / MFMA | `tl.dot` | `QK^T`, `PV` の高速化 |
| on-chip memory | register/shared memory | compiler-managed block values | `m`, `l`, `acc`, Q/K/V tile |

## 最低限の暗記項目

- kernel は GPU 上で動く関数。
- launch は CPU が GPU に kernel 実行を投入する操作。
- CUDA は thread/block/grid、Triton は program/tile/grid を主に見る。
- NVIDIA の warp は 32 lane。
- `tl.program_id` は program instance の ID。
- `tl.arange` は tile 内 offset vector。
- `num_warps` は 1 program に使う warp 数のヒント。
- `tl.dot` は条件が合うと Tensor Core / MFMA 系の行列命令に lower される。
- FlashAttention は「1 program = 1 query block」が基本形。

## 参照

- NVIDIA CUDA Programming Guide: Programming Model, Warps and SIMT, GPU Memory.
- Triton language API: `program_id`, `load`, `store`, `dot`.
- OpenAI Triton introduction: block programming model and compiler-managed optimizations.

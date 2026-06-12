# kernel, warp, occupancy, scheduling

この章では、`num_warps` を単なるチューニング値ではなく、SM 上の資源配分として理解します。FlashAttention では、`num_warps` を増やせば速くなるとは限りません。1 program 内の並列度は上がりますが、1 SM に同時常駐できる program 数が減り、memory latency を隠しにくくなることがあります。

## SM と resident program

GPU は複数の SM、Streaming Multiprocessor、から構成されます。kernel launch によって生成された work unit は SM に割り当てられます。CUDA では thread block、Triton では program instance がこの単位に近いです。

1 SM には複数の block/program が同時に常駐できます。ただし上限は以下で制約されます。

- SM あたり最大 thread 数
- SM あたり最大 block/program 数
- SM あたり最大 warp 数
- register file 容量
- shared memory / SRAM 容量
- barrier や pipeline stage などの内部資源

Triton でまず見るべき meta-parameters は次です。

```text
BLOCK_M, BLOCK_N, BLOCK_D
num_warps
num_stages
```

`BLOCK_*` が大きいほど 1 program あたりの計算量と reuse は増えますが、register/shared memory の使用量も増えます。`num_warps` が大きいほど 1 program 内の並列度は増えますが、resident programs per SM が減る可能性があります。

## occupancy の意味

occupancy は、SM が同時に保持できる最大 warp 数に対して、実際に active になれる warp 数の割合です。

この式は、理論上の active warp 利用率を表します。

\[
\mathrm{occupancy}
=
\frac{\mathrm{active\ warps\ per\ SM}}
{\mathrm{maximum\ resident\ warps\ per\ SM}}
\]

これは「GPU が何 % のピーク性能を出すか」ではありません。高 occupancy でも memory access が悪ければ遅く、低 occupancy でも Tensor Core を高効率に使えていれば速いことがあります。

## `num_warps` の近似計算

NVIDIA GPU では warp size は 32 です。したがって、Triton の `num_warps=4` は概念上次のように考えられます。

\[
\mathrm{threads\ per\ program}
\approx
\mathrm{num\_warps} \times 32
\]

`max_threads_per_sm = 2048` の GPU であれば、thread 数だけを見た resident program 上限は次のようになります。

\[
\mathrm{resident\ programs\ by\ threads}
=
\left\lfloor
\frac{2048}{32 \cdot \mathrm{num\_warps}}
\right\rfloor
\]

たとえば `num_warps=4` なら、thread 数だけでは最大 16 program が常駐可能です。しかし実際には、最大 block 数、register、shared memory、compiler が選んだ schedule によってさらに下がります。

## なぜ occupancy だけでは決められないか

FlashAttention は次を同時に満たす必要があります。

1. `QK^T` と `PV` で Tensor Core / MFMA を使う。
2. Q tile を再利用する。
3. K/V tile の HBM load を coalesced にする。
4. online softmax の `m`, `l`, `acc` を register に保持する。
5. register pressure が高すぎて spill しない。
6. 1 SM に十分な program が常駐して memory latency を隠す。

`BLOCK_M=128, BLOCK_N=128, HEAD_DIM=128` のような大きな tile は計算密度が高い一方、accumulator が大きく、register pressure が上がります。`BLOCK_M=16` のように小さすぎる tile は occupancy は高くても、Tensor Core の効率や data reuse が落ちやすくなります。

## warp divergence と mask

warp divergence は、同じ warp 内の lane が異なる branch を通ることです。Triton では per-thread branch を直接書く機会は少ないですが、mask はよく使います。

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

この mask は境界処理には必要ですが、常に半分だけ有効なような tile 形状を選ぶと無駄が増えます。FlashAttention の causal mask では、query block と key block の位置関係によって完全に有効な block、完全に無効な block、三角 mask が必要な境界 block が分かれます。高速化では、完全無効 block を skip し、完全有効 block では mask を減らす分岐が重要です。

## lesson で確認すること

`lessons/01_gpu_execution_model.py` では、`program_id` がどの要素を担当するかを実際に tensor に書き出します。`lessons/02_warps_sms_occupancy.py` では、device properties から `num_warps` ごとの粗い occupancy 上限を表示します。

この章のゴールは、次の問いに答えられることです。

- この Triton kernel の 1 program は何を担当しているか。
- `grid` の各 axis はどの tensor dimension に対応しているか。
- `num_warps` を増やすと、何が速くなり、何が悪化しうるか。
- causal FlashAttention で mask が多い block はどこか。
- occupancy が低いとき、それは本当に問題か、それとも Tensor Core 利用率が十分高いか。

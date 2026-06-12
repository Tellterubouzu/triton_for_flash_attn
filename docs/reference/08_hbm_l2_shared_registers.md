# 08. HBM, L2, shared memory, registers, occupancy

この章の目的は、kernel が遅い理由を「FLOPs が足りない」ではなく「HBM bandwidth、cache hit、register pressure、occupancy、memory coalescing」の語彙で説明できるようにすることです。

## どの値を取得できるか

Python から比較的安定して取得できるもの:

- HBM 総量・空き容量: `torch.cuda.mem_get_info()`
- device properties: `torch.cuda.get_device_properties()`
- PyTorch allocator の allocated/reserved: `torch.cuda.memory_allocated()`, `torch.cuda.memory_reserved()`
- tensor の pointer/stride/storage offset: `tensor.data_ptr()`, `tensor.stride()`, `tensor.storage_offset()`

取得しにくい、または profiler に任せるべきもの:

- 実際の L1/L2 hit rate
- shared memory bank conflict
- register spill
- warp stall reason
- Tensor Core utilization

これらは Nsight Compute、rocprof、または vendor profiler の counter を見るのが基本です。

## roofline の最小モデル

次の式は、kernel の arithmetic intensity を表します。

\[
I = \frac{\mathrm{FLOPs}}{\mathrm{HBM\ bytes}}
\]

`I` が小さい kernel は memory-bound になりやすく、`I` が大きい kernel は compute-bound になりやすいです。Vector add は 1 element あたり read 2 個 + write 1 個なので、fp32 では 12 byte に対して 1 FLOP 程度しかありません。MatMul は tile 再利用が効けば FLOPs/byte が大きくなります。

FlashAttention は naive attention より中間行列の HBM read/write を減らすので、同じ数式でも実効的な bytes が大きく変わります。

## register と occupancy

Triton で `BLOCK_M`, `BLOCK_N`, `BLOCK_D` を大きくすると、1 program が保持する block tensor と accumulator が大きくなります。これは data reuse を増やしますが、register 使用量や shared memory 使用量を増やし、同時に resident できる program 数を減らす可能性があります。

最初に見るべき knobs:

- `BLOCK_SIZE` for elementwise
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K` for matmul/attention
- `num_warps`
- `num_stages`
- cache hint / eviction policy

## Triton で直接できること・できないこと

できること:

- HBM/global memory から block load/store する。
- mask と boundary check を使って out-of-bound を避ける。
- cache modifier / eviction policy を指定する。
- tile shape と pipeline stages を変える。
- generated IR/PTX を保存し、compiler が何を生成したか見る。

基本的にはできないこと:

- CUDA C++ のように shared memory 配列を直接宣言して手動で layout する。
- HBM allocation を Triton kernel 内で malloc/free する。
- GPU cache を一般的な意味で「消去」する。
- どの load が必ず L1 に入るかを portable に保証する。

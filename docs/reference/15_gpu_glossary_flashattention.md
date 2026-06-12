# FlashAttention kernel を読むための GPU 用語集

## kernel

GPU 上で実行される関数。Triton では `@triton.jit` 関数。CPU から launch される。

## launch

CPU が GPU に kernel 実行を投入する操作。小さい kernel を大量に launch すると overhead が支配的になることがある。

## grid

kernel 全体の実行空間。Triton では `kernel[grid](...)` の `grid`。FlashAttention では query block axis と batch-head axis を置くことが多い。

## program instance

Triton の主要な実行単位。CUDA の thread block / CTA に近い。`tl.program_id(axis)` で ID を取る。

## tile

1 program が処理するデータ block。たとえば `Q[BLOCK_M, HEAD_DIM]` や `K[BLOCK_N, HEAD_DIM]`。

## warp

NVIDIA GPU の SIMT 実行単位。32 lanes。Triton では `num_warps` で 1 program に使う warp 数を指定する。

## wavefront

AMD GPU における warp に近い概念。サイズは世代やモードに依存するため、NVIDIA の warp size 32 をそのまま仮定しない。

## SM / CU

NVIDIA では Streaming Multiprocessor、AMD では Compute Unit。block/program が実行される主要な演算資源単位。

## occupancy

SM/CU が同時に保持できる warp/wave に対して、active になれる warp/wave の割合。高ければ常に速いわけではない。

## register

thread/program 内の最速級の一時保存領域。Triton の scalar/block value は compiler によって register に置かれることが多い。使いすぎると register pressure が上がり、occupancy 低下や spill の原因になる。

## shared memory / SRAM

SM 内の programmable on-chip memory。CUDA では明示的に `__shared__` を使う。Triton では多くの場合 compiler が `tl.dot` などから shared memory 利用を決める。

## HBM / global memory

GPU に接続された大容量 DRAM。容量は大きいが on-chip memory より遅い。FlashAttention は HBM に巨大な attention matrix を書かないことで速くなる。

## L1 / L2 cache

global memory access を cache する階層。L1 は SM 近傍、L2 は GPU 全体で共有される。Triton では `tl.load(..., cache_modifier=...)` などの hint を試せる。

## memory coalescing

隣接 lane/program が連続 address を読むことで、大きな memory transaction にまとめられること。stride access が悪いと bandwidth が落ちる。

## Tensor Core

NVIDIA GPU の行列積累算用専用ユニット。Triton では主に `tl.dot` 経由で使う。対応 dtype と tile shape は GPU 世代に依存する。

## MFMA / Matrix Core

AMD GPU の行列演算命令/ユニット系。Triton の `tl.dot` が backend に応じて lower される。

## MMA

Matrix Multiply-Accumulate。小さな matrix tile について `D = A @ B + C` を行う演算。Tensor Core や MFMA の中心。

## `tl.dot`

Triton の block-level matrix multiplication。FlashAttention では `QK^T` と `PV` の両方で使う。

## `num_warps`

Triton program instance に割り当てる warp 数。大きい tile/reduction/dot では増やすと速くなる場合があるが、resident program 数が減ることもある。

## `num_stages`

Triton の software pipelining stage 数。load と compute の重なりに影響する。大きいほどよいとは限らず、register/shared memory 使用量が増えることがある。

## warp divergence

同じ warp 内の lane が異なる branch を通ることで一部 lane が mask され、効率が落ちる現象。Triton では明示 thread branch は少ないが、mask と causal boundary で類似の無駄が出る。

## arithmetic intensity

1 byte の memory traffic あたり何 FLOP するか。FlashAttention は HBM temporary を減らすことで実効 arithmetic intensity を上げる。

## roofline

演算性能上限と memory bandwidth 上限から、kernel が compute-bound か memory-bound かを見るモデル。

## spill

register に収まらない一時変数が local memory に退避されること。local memory は実体として global memory 側に置かれることがあり、非常に遅くなる。

## causal mask

autoregressive attention で未来 token を見ないための mask。FlashAttention では query block と key block の位置関係に応じて skip/full/boundary を分けると効率が上がる。

## persistent kernel

work queue や tile scheduling を工夫し、SM に常駐するように設計された kernel。高性能 attention/matmul で使われることがあるが、教材では発展課題。

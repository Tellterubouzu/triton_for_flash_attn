# 01. GPU Kernel の性能モデル

Triton kernel を書く前に、対象演算がなぜ遅いかを分類します。主な分類は memory-bound、compute-bound、launch-bound です。

## Memory-bound

memory-bound とは、計算器ではなく HBM との読み書き帯域が支配的な状態です。Vector Add は典型例です。

この数式は、2 つのベクトルを足して 1 つのベクトルへ保存する操作を表します。

\[
c_i = a_i + b_i
\]

各要素について 2 load + 1 store なので、float32 なら 12 bytes に対して加算 1 FLOP です。arithmetic intensity は約 \(1/12\) FLOP/byte で、計算器を使い切る前に memory bandwidth が詰まります。

## Compute-bound

compute-bound とは、HBM よりも演算器が支配的な状態です。大きい matmul は典型例です。

この数式は、行列積を表します。

\[
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}
\]

\(M=N=K=4096\) の matmul は膨大な FLOPs を持ち、tile reuse により memory access あたりの計算量が大きくなります。PyTorch の `matmul` は vendor library を使うため非常に強い baseline です。単体 matmul を自作する理由は通常ありません。自作する理由は、matmul の前後の処理を融合したい場合です。

## Launch-bound

launch-bound とは、演算量が小さい kernel を大量に launch するため、GPU kernel launch overhead や Python overhead が支配的な状態です。小さい elementwise chain や小 batch の推論で起きやすいです。

この場合は、Triton より前に `torch.compile`、operator fusion、batching、CUDA graph を検討します。

## FlashAttention の位置づけ

通常の attention は、\(QK^\top\) と softmax probability \(P\) を HBM に保存します。

\[
O=PV,\quad P=\mathrm{softmax}(QK^\top/\sqrt{d})
\]

この実装は FLOPs も大きいですが、長い sequence では \([B,H,N,N]\) の中間行列が memory footprint と HBM traffic の支配要因になります。FlashAttention は exact attention のまま、中間行列を HBM に materialize しないように tile 化します。

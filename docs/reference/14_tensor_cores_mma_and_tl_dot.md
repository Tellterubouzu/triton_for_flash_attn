# Tensor Core, MMA, `tl.dot`

この章では、Tensor Core を「なんとなく速い行列演算器」ではなく、FlashAttention kernel の中心的な制約として扱います。FlashAttention forward の重い計算は主に次の 2 つです。

この数式は、query block と key block から attention score block を計算する行列積を表します。

\[
S_{BM \times BN}
=
Q_{BM \times D} K^\top_{D \times BN}
\]

この数式は、softmax probability block と value block から出力 accumulator を更新する行列積を表します。

\[
O_{BM \times D}
\leftarrow
P_{BM \times BN} V_{BN \times D}
\]

どちらも小さな tile 行列積です。この部分が Tensor Core / MFMA に乗るかどうかで性能が大きく変わります。

## Tensor Core とは何か

Tensor Core は、NVIDIA GPU に搭載される行列積累算用の専用演算ユニットです。通常の FP32 scalar/vector ALU で要素ごとに multiply-add するのではなく、小さな行列 tile に対する matrix multiply-accumulate、MMA、を高スループットで実行します。

概念的には、次の演算を専用命令で行います。

\[
D = A B + C
\]

ここで、A と B は FP16/BF16/TF32/FP8/INT8 などの低精度入力、C と D は FP32 accumulator などの組み合わせになります。対応 dtype は GPU 世代と backend に依存します。

AMD CDNA 系 GPU では、NVIDIA Tensor Core という名称ではなく、Matrix Core / MFMA 命令系として似た役割の行列専用演算が使われます。Triton では `tl.dot` を書くことで、backend が対応する行列命令へ lower できる場合があります。

## `tl.dot` の意味

Triton の `tl.dot(a, b)` は、block tensor 同士の matrix multiplication を表す高レベル演算です。

```python
a = tl.load(a_ptrs)  # shape: (BLOCK_M, BLOCK_K)
b = tl.load(b_ptrs)  # shape: (BLOCK_K, BLOCK_N)
acc += tl.dot(a, b)
```

`tl.dot` は単なる Python の for-loop ではありません。compiler は tile shape、dtype、target backend を見て、Tensor Core / MFMA などに lower できるかを判断します。

Tensor Core に乗りやすい条件は典型的には次です。

- 入力 dtype が FP16, BF16, TF32, FP8, INT8 など対象 GPU でサポートされる形式。
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K` が backend の MMA tile に合う。
- memory layout と stride が効率的。
- `tl.dot` の前後で不要な型変換や非連続 access が多すぎない。

## TF32 と `input_precision`

NVIDIA Ampere 以降では、FP32 入力の matrix multiply に TF32 Tensor Core を使うことがあります。Triton の `tl.dot` には `input_precision` があります。

```python
acc += tl.dot(a, b, input_precision="tf32")
```

`"tf32"` は FP32 入力を TF32 相当で Tensor Core に乗せる設定です。`"ieee"` はより厳密な FP32 演算を選びますが、性能は下がる可能性があります。FlashAttention では通常 FP16/BF16 入力を FP32 accumulator で扱うことが多いため、まずは FP16/BF16 の `tl.dot` が対象になります。

## Tensor Core 利用をどう確認するか

最も確実なのは profiler です。Nsight Compute で Tensor Core 系 metric、SM pipe utilization、MMA instruction count を見るのが本筋です。ただし、教材ではまず生成コードを軽く見る方法を入れています。

`lessons/14_tensor_cores_and_tl_dot.py` は、最小 `tl.dot` kernel を compile し、取得できる asm / PTX / LLVM / TTGIR から次のような token を探します。

```text
NVIDIA: mma, wgmma, ldmatrix, cp.async
AMD    : mfma, dot
Triton : tt.dot, dot
```

注意点として、PTX や IR に `mma` が見えないから必ず Tensor Core が使われていない、とは断定できません。compiler stage、backend、driver、Triton version によって表示される representation は変わります。最終判断は profiler metric で行います。

## FlashAttention の tile shape と Tensor Core

FlashAttention で Tensor Core を使いたい場合、head dimension `D` と `BLOCK_N` は特に重要です。

```text
Q: (BLOCK_M, D)
K: (BLOCK_N, D)
S = Q @ K.T: (BLOCK_M, BLOCK_N)

P: (BLOCK_M, BLOCK_N)
V: (BLOCK_N, D)
O = P @ V: (BLOCK_M, D)
```

`D=64` や `D=128` は Tensor Core に載せやすい典型値です。一方で、`D` が奇妙な値、stride が非連続、`BLOCK_N` が小さすぎる、mask が過剰、dtype が FP32 strict などの場合、期待した行列命令効率が出ないことがあります。

## accumulator と精度

FP16/BF16 入力でも、accumulator は FP32 にするのが一般的です。

この数式は、低精度入力を使いつつ累積は高精度で行う混合精度行列積を表します。

\[
C_{ij}^{\mathrm{fp32}}
=
\sum_k
\mathrm{cast}_{\mathrm{fp32}}(A_{ik})
\mathrm{cast}_{\mathrm{fp32}}(B_{kj})
\]

FlashAttention ではさらに softmax が入るため、score の max、分母 `l`、output accumulator の dtype が数値安定性に影響します。高速化のためにすべてを低精度に落とすと、長い sequence や大きな logit で誤差が出やすくなります。

## Tensor Core は万能ではない

Tensor Core は行列積には強いですが、FlashAttention kernel 全体には次の非 matmul 処理もあります。

- score scaling
- causal mask / padding mask
- row-wise max reduction
- exp
- row-wise sum reduction
- online softmax の rescale
- store/load の address 計算

FlashAttention-2 が重視した点の 1 つは、Tensor Core でない非 matmul FLOPs を減らし、warp 間の work partitioning を改善することです。つまり、`tl.dot` を使うだけでは不十分で、dot の周辺処理をどれだけ軽くするかが重要です。

## 実験課題

1. `lessons/14_tensor_cores_and_tl_dot.py --dtype fp16` を実行し、asm summary に `mma` / `wgmma` / `mfma` などが出るか確認する。
2. `--dtype fp32 --input-precision tf32` と `--dtype fp32 --input-precision ieee` を比較する。
3. `--block-m`, `--block-n`, `--block-k` を変え、生成コードと速度を比較する。
4. `benchmarks/bench_num_warps.py` で `num_warps` を変え、matmul-like workload の速度を見る。
5. Nsight Compute で Tensor Core utilization を確認する。

## 参照

- NVIDIA CUDA Programming Guide: Compute Capabilities and Tensor Core input data types.
- Triton API: `triton.language.dot`.
- OpenAI Triton introduction: block-level operations and compiler-managed shared memory / scheduling.

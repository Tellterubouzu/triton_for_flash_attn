# References

## Triton

- Triton documentation: installation, tutorials, Python API.
- Triton GitHub: Compatibility、debugging env vars、source build。
- Triton MAPL 2019: Philippe Tillet, H. T. Kung, David Cox, “Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.”

Triton MAPL 2019 の手法は、tile を中心とする中間言語と compiler pass により、tensor program を GPU code へ落とすものです。結果として、matrix multiplication や convolution で hand-tuned vendor library に近い portability/performance を狙えることを示しています。限界は、現在の Python-first Triton API、MLIR backend、最新 GPU 世代、FlashAttention 系 kernel の実装詳細をそのまま説明する資料ではない点です。

## PyTorch

- PyTorch profiler recipe: operator の time と memory consumption を測る手順。
- `torch.compile` tutorial: graph tracing、graph break、speedup の確認。
- User-defined Triton kernels with `torch.compile`: custom Triton kernel を PyTorch graph に統合する入口。

## FlashAttention

- Tri Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” 2022.

この論文の手法は、exact attention を保ったまま、tiling と online softmax により HBM と on-chip SRAM 間の read/write を減らすことです。報告結果として、BERT-large で MLPerf 1.1 training record に対して 15% end-to-end wall-clock speedup、GPT-2 sequence length 1K で 3x、Long Range Arena sequence length 1K-4K で 2.4x の speedup が示されています。限界は、attention の計算量そのものは sequence length に対して二次のままであり、主な改善対象は IO complexity と memory footprint である点です。

- Tri Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning,” 2023.

この論文の手法は、FlashAttention の非 matmul FLOPs を減らし、single head 内でも thread block 間で並列化し、warp 間の work partitioning を改善することです。報告結果として、FlashAttention に対して約 2x の speedup、A100 上で理論最大 FLOPs の 50-73% への到達、GPT-style model training で 225 TFLOP/s per A100、72% model FLOPs utilization が示されています。限界は、GPU 世代や head_dim、causal/non-causal、backward、実装依存の scheduling に性能が大きく依存する点です。


## Low-level memory and profiling references

- NVIDIA CUDA Runtime API: `cudaDeviceGetAttribute`, `cudaMemGetInfo`, device properties, cache configuration, stream synchronization.
- NVIDIA CUDA C Programming Guide: memory hierarchy, global/shared/local/register memory, asynchronous execution, streams.
- Triton Python API: `tl.load`, `tl.store`, `tl.make_block_ptr`, `tl.advance`, cache modifiers, eviction policy, block pointer semantics.
- PyTorch CUDA memory API: `torch.cuda.mem_get_info`, `torch.cuda.get_device_properties`, `torch.cuda.memory_allocated`, `torch.cuda.memory_reserved`, `torch.cuda.empty_cache`.
- PyTorch Triton compilation stages blog: TTIR, TTGIR, LLVM IR, PTX extraction and interpretation.

これらは、Triton source で書いた `tl.load/tl.store/tl.dot` が backend でどのように lowering され、どの memory hierarchy に負荷をかけるかを見るための資料です。限界は、実際の cache hit rate、register spill、shared memory bank conflict などは GPU 世代・driver・compiler に依存し、最終的には profiler counter で確認する必要がある点です。

## GPU execution model / warp / Tensor Core 追加参照

- NVIDIA CUDA Programming Guide, Programming Model. Kernel launch、thread block、grid、warp/SIMT、GPU memory の定義を確認する一次資料。
  - https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
- NVIDIA CUDA Programming Guide, Compute Capabilities. warp size、resident warps、shared memory、Tensor Core 対応 dtype などの世代別表を確認する一次資料。
  - https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- Triton language API. `program_id`, `load`, `store`, `dot` などの API 定義。
  - https://triton-lang.org/main/python-api/triton.language.html
  - https://triton-lang.org/main/python-api/generated/triton.language.dot.html
- OpenAI Triton introduction. Triton の block programming model と compiler-managed optimization の背景。
  - https://openai.com/index/triton/

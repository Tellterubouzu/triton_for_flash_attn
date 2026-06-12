# Glossary

この用語集は、module docs の「新出用語」を統合したものです。`初出` は、その用語を最初に重点的に扱う module を示します。

| 用語 | 定義 | 初出 |
|---|---|---|
| GPU | 多数の演算器を持つ並列計算装置。Triton kernel は主に GPU 上で実行される。 | [00](modules/00_setup_and_sanity.md) |
| CUDA | NVIDIA GPU 向けのプログラミング環境と runtime。 | [00](modules/00_setup_and_sanity.md) |
| ROCm | AMD GPU 向けの計算ソフトウェアスタック。 | [00](modules/00_setup_and_sanity.md) |
| Triton | Python から GPU kernel を記述し、GPU 向けコードへコンパイルする言語・コンパイラ。 | [00](modules/00_setup_and_sanity.md) |
| kernel | GPU 上で多数の並列実行単位として走る関数。CPU の通常関数呼び出しとは実行モデルが異なる。 | [00](modules/00_setup_and_sanity.md) |
| synchronization | CPU が GPU の非同期処理の完了を待つこと。benchmark では必須。 | [00](modules/00_setup_and_sanity.md) |
| warmup | 初回コンパイル、allocator 初期化、cache 状態などを本計測から除くための予備実行。 | [00](modules/00_setup_and_sanity.md) |
| launch | CPU 側から GPU kernel の実行を開始する操作。launch 自体にも固定 overhead がある。 | [01](modules/01_gpu_execution_model.md) |
| grid | kernel launch で作られる実行単位全体の集合。Triton では grid が program instance 数を決める。 | [01](modules/01_gpu_execution_model.md) |
| thread block | CUDA で 1 つの SM に配置される thread のまとまり。Triton では直接 threadIdx を書かず program 単位で考える。 | [01](modules/01_gpu_execution_model.md) |
| Triton program instance | Triton kernel の 1 つの実行単位。`tl.program_id` でどの instance かを取得する。 | [01](modules/01_gpu_execution_model.md) |
| tile | 1 program がまとめて処理する連続または矩形のデータブロック。 | [01](modules/01_gpu_execution_model.md) |
| warp | NVIDIA GPU で 32 threads が SIMT として一緒に命令を実行する単位。 | [01](modules/01_gpu_execution_model.md) |
| SIMT | Single Instruction, Multiple Threads。複数 thread が同じ命令列を異なるデータに適用する実行方式。 | [01](modules/01_gpu_execution_model.md) |
| SM | Streaming Multiprocessor。GPU の主要な実行ユニットで、warp scheduler、register file、shared memory などを持つ。 | [02](modules/02_warps_sms_occupancy.md) |
| occupancy | SM に同時滞在できる warp/thread/program の割合。高ければ常に速いわけではない。 | [02](modules/02_warps_sms_occupancy.md) |
| resident program | ある時点で SM 上に配置されている program instance。 | [02](modules/02_warps_sms_occupancy.md) |
| warp scheduler | 実行可能な warp を選んで命令を発行するハードウェア。 | [02](modules/02_warps_sms_occupancy.md) |
| register pressure | kernel が必要とする register 数の多さ。増えすぎると occupancy 低下や spill につながる。 | [02](modules/02_warps_sms_occupancy.md) |
| num_warps | Triton の meta-parameter。1 program の中で使う warp 数の指定。 | [02](modules/02_warps_sms_occupancy.md) |
| num_stages | Triton の pipelining 段数を指定する meta-parameter。主に load と compute の重なりに関係する。 | [02](modules/02_warps_sms_occupancy.md) |
| HBM | High Bandwidth Memory。GPU board 上の大容量 global memory。FlashAttention では HBM read/write 削減が主目標になる。 | [03](modules/03_memory_hierarchy_and_device_query.md) |
| L2 cache | SM 群で共有される cache。HBM より速く小さい。 | [03](modules/03_memory_hierarchy_and_device_query.md) |
| SRAM | GPU chip 上の高速小容量メモリの総称。shared memory や register file を説明するときに使う。 | [03](modules/03_memory_hierarchy_and_device_query.md) |
| shared memory | SM 内で thread block/program が使う高速な on-chip memory。Triton では明示配列ではなく tile と compiler 管理として現れることが多い。 | [03](modules/03_memory_hierarchy_and_device_query.md) |
| register file | 各 thread/program の一時値を保持する最も近い記憶資源。 | [03](modules/03_memory_hierarchy_and_device_query.md) |
| bandwidth | 単位時間あたりに転送できる byte 数。 | [03](modules/03_memory_hierarchy_and_device_query.md) |
| latency | 1 回の memory access の開始から完了までの遅延。 | [03](modules/03_memory_hierarchy_and_device_query.md) |
| pointer | memory 上の位置を指す値。Triton の `ptr + offsets` は要素単位の pointer arithmetic として扱う。 | [04](modules/04_tensor_addresses_strides_layouts.md) |
| byte address | byte 単位で数えた memory address。 | [04](modules/04_tensor_addresses_strides_layouts.md) |
| data_ptr | PyTorch Tensor の先頭要素付近を指す pointer 値を取得する API。 | [04](modules/04_tensor_addresses_strides_layouts.md) |
| storage | Tensor の実データを保持する memory 領域。view は同じ storage を共有できる。 | [04](modules/04_tensor_addresses_strides_layouts.md) |
| storage_offset | Tensor の論理 index 0 が storage の何要素目から始まるかを表す offset。 | [04](modules/04_tensor_addresses_strides_layouts.md) |
| stride | 各次元で index を 1 増やしたときに進む element 数。byte 数ではない。 | [04](modules/04_tensor_addresses_strides_layouts.md) |
| contiguous | 標準的な row-major layout で隙間なく並んでいる状態。 | [04](modules/04_tensor_addresses_strides_layouts.md) |
| effective bandwidth | 実測時間と論理的な転送 byte 数から計算した帯域。理論帯域とは異なる。 | [05](modules/05_hbm_bandwidth_and_copy.md) |
| read traffic | HBM から読み出す byte 数。 | [05](modules/05_hbm_bandwidth_and_copy.md) |
| write traffic | HBM へ書き込む byte 数。 | [05](modules/05_hbm_bandwidth_and_copy.md) |
| memory-bound | 実行時間が演算器ではなく memory bandwidth に主に制限される状態。 | [05](modules/05_hbm_bandwidth_and_copy.md) |
| copy kernel | 入力を読み、出力へ書く最小の memory transfer kernel。 | [05](modules/05_hbm_bandwidth_and_copy.md) |
| coalescing | 近い address への複数 access を少数の memory transaction にまとめること。 | [06](modules/06_coalescing_strides_cache.md) |
| memory transaction | GPU memory subsystem が実際に処理する転送単位。 | [06](modules/06_coalescing_strides_cache.md) |
| cache line/sector | cache がまとめて保持・転送する memory 範囲。 | [06](modules/06_coalescing_strides_cache.md) |
| stride access | 隣接要素ではなく一定間隔で memory を読む access pattern。 | [06](modules/06_coalescing_strides_cache.md) |
| cache hit | 必要な data が cache に存在し、HBM まで行かずに読めること。 | [06](modules/06_coalescing_strides_cache.md) |
| locality | 近い時間または近い address の data を再利用できる性質。 | [06](modules/06_coalescing_strides_cache.md) |
| tl.program_id | 現在の Triton program instance の index を返す関数。 | [07](modules/07_vector_add_first_kernel.md) |
| tl.arange | tile 内の vector offset を作る Triton primitive。 | [07](modules/07_vector_add_first_kernel.md) |
| tl.load | pointer から data を読み込む操作。mask と other で範囲外を安全に処理できる。 | [07](modules/07_vector_add_first_kernel.md) |
| tl.store | pointer へ data を書き込む操作。mask で範囲外 write を防ぐ。 | [07](modules/07_vector_add_first_kernel.md) |
| mask | 有効な lane/要素だけを load/store/計算するための boolean tensor。 | [07](modules/07_vector_add_first_kernel.md) |
| BLOCK_SIZE | 1 program が処理する要素数を表す compile-time constant。 | [07](modules/07_vector_add_first_kernel.md) |
| fusion | 複数の演算を 1 kernel にまとめ、中間 tensor の HBM read/write を消す最適化。 | [08](modules/08_elementwise_fusion.md) |
| intermediate tensor | 演算の途中結果として materialize される tensor。 | [08](modules/08_elementwise_fusion.md) |
| arithmetic intensity | 転送 byte あたりの FLOPs。低いほど memory-bound になりやすい。 | [08](modules/08_elementwise_fusion.md) |
| launch overhead | kernel 実行を開始する固定コスト。小さい演算では支配的になりやすい。 | [08](modules/08_elementwise_fusion.md) |
| cache modifier | load/store が cache をどう使うかを compiler/backend に伝える hint。 | [09](modules/09_cache_hints_and_eviction.md) |
| eviction policy | cache に長く残したいか早く追い出してよいかを伝える hint。 | [09](modules/09_cache_hints_and_eviction.md) |
| .ca | NVIDIA PTX の load cache modifier 例。通常の cache-all 的な意味で使われる。 | [09](modules/09_cache_hints_and_eviction.md) |
| .cg | global/L2 寄りの cache 方針を示す modifier。 | [09](modules/09_cache_hints_and_eviction.md) |
| .cv | volatile に近い再読込を促す modifier。 | [09](modules/09_cache_hints_and_eviction.md) |
| hint | 性能上の希望を伝える指定。意味や効果は backend/GPU/compiler に依存する。 | [09](modules/09_cache_hints_and_eviction.md) |
| block pointer | base pointer、shape、stride、offset、block shape をまとめた Triton の tile pointer。 | [10](modules/10_block_pointers_boundary_check.md) |
| tl.make_block_ptr | block pointer を作る Triton API。 | [10](modules/10_block_pointers_boundary_check.md) |
| boundary_check | block pointer load/store で境界外次元を自動的に処理する指定。 | [10](modules/10_block_pointers_boundary_check.md) |
| padding_option | 境界外 load に zero などを入れる指定。 | [10](modules/10_block_pointers_boundary_check.md) |
| tl.advance | block pointer の offsets を進めて次 tile を指す操作。 | [10](modules/10_block_pointers_boundary_check.md) |
| tensor descriptor | tensor の形状・stride・layout を kernel に伝える情報のまとまり。 | [10](modules/10_block_pointers_boundary_check.md) |
| reduction | 複数要素から max/sum など 1 つまたは少数の値を作る演算。 | [11](modules/11_reductions_and_softmax.md) |
| row-wise | 行ごとに独立して計算すること。softmax や LayerNorm で頻出。 | [11](modules/11_reductions_and_softmax.md) |
| stable softmax | overflow を避けるため各行の最大値を引いてから exp を取る softmax。 | [11](modules/11_reductions_and_softmax.md) |
| max trick | softmax の不変性を使い、logit から最大値を引く数値安定化。 | [11](modules/11_reductions_and_softmax.md) |
| power-of-two block | Triton の reduction を扱いやすくするため、列数を 2 の冪へ丸めた block size。 | [11](modules/11_reductions_and_softmax.md) |
| fp32 | 32-bit floating point。accumulator や reference 計算で使う標準的な精度。 | [12](modules/12_numerics_and_correctness.md) |
| fp16 | 16-bit floating point。高速だが表現範囲と精度が狭い。 | [12](modules/12_numerics_and_correctness.md) |
| bf16 | bfloat16。fp16 より mantissa は粗いが exponent 範囲が fp32 に近い。 | [12](modules/12_numerics_and_correctness.md) |
| rounding | 実数を有限精度浮動小数へ丸めること。 | [12](modules/12_numerics_and_correctness.md) |
| absolute error | 予測値と基準値の絶対差。 | [12](modules/12_numerics_and_correctness.md) |
| relative error | 基準値の大きさに対する誤差。 | [12](modules/12_numerics_and_correctness.md) |
| NaN/Inf | 未定義値や無限大。kernel 検証で必ず検出するべき異常値。 | [12](modules/12_numerics_and_correctness.md) |
| GEMM | General Matrix-Matrix Multiplication。C = A @ B の一般的な行列積。 | [13](modules/13_tiled_matmul.md) |
| K loop | matmul の内積次元 K を BLOCK_K ごとに走査する loop。 | [13](modules/13_tiled_matmul.md) |
| accumulator | 部分和を保持する一時領域。通常 fp32 で持つ。 | [13](modules/13_tiled_matmul.md) |
| SRAM reuse | on-chip 側に読んだ tile を複数演算で使い回し、HBM access を減らすこと。 | [13](modules/13_tiled_matmul.md) |
| grouped ordering | program の実行順を並べ替え、L2 reuse を改善する matmul の scheduling 技法。 | [13](modules/13_tiled_matmul.md) |
| Tensor Core | NVIDIA GPU の行列積専用演算器。fp16/bf16/tf32 等の matrix multiply-accumulate を高スループットで実行する。 | [14](modules/14_tensor_cores_and_tl_dot.md) |
| MMA | Matrix Multiply-Accumulate。小さな行列ブロックの積和命令。 | [14](modules/14_tensor_cores_and_tl_dot.md) |
| MFMA | AMD GPU の matrix fused multiply-add 系命令。 | [14](modules/14_tensor_cores_and_tl_dot.md) |
| tl.dot | Triton の block 行列積 primitive。FlashAttention の QK^T と PV で中核になる。 | [14](modules/14_tensor_cores_and_tl_dot.md) |
| TF32 | NVIDIA Ampere 以降で fp32 入力を Tensor Core 向けに扱う形式。 | [14](modules/14_tensor_cores_and_tl_dot.md) |
| input_precision | `tl.dot` で fp32 入力の精度方針などを指定する引数。 | [14](modules/14_tensor_cores_and_tl_dot.md) |
| lowering | Triton の高水準表現を LLVM/PTX/ISA など低水準表現へ変換すること。 | [14](modules/14_tensor_cores_and_tl_dot.md) |
| LayerNorm | サンプル内の特徴次元で平均と分散を計算し正規化する層。 | [15](modules/15_layernorm_and_rowwise_norm.md) |
| mean | 平均。LayerNorm では行内特徴量の中心を表す。 | [15](modules/15_layernorm_and_rowwise_norm.md) |
| variance | 分散。特徴量のばらつきを表す。 | [15](modules/15_layernorm_and_rowwise_norm.md) |
| epsilon | 分母が 0 に近づくことを避けるために分散へ足す小さい値。 | [15](modules/15_layernorm_and_rowwise_norm.md) |
| affine | 正規化後に scale と bias を適用する線形変換。 | [15](modules/15_layernorm_and_rowwise_norm.md) |
| saved stats | backward で再利用する mean/rstd などの統計量。 | [15](modules/15_layernorm_and_rowwise_norm.md) |
| profiler | 実行時間、memory、呼び出し回数などを記録する測定ツール。 | [16](modules/16_bottleneck_lab_and_profiler.md) |
| hotspot | 全体時間への寄与が大きい箇所。 | [16](modules/16_bottleneck_lab_and_profiler.md) |
| operator | PyTorch の `aten::matmul` などフレームワーク上の演算単位。 | [16](modules/16_bottleneck_lab_and_profiler.md) |
| CUDA/HIP kernel time | 実際に GPU 上で動いた kernel の実行時間。 | [16](modules/16_bottleneck_lab_and_profiler.md) |
| trace | 時間軸上の CPU/GPU activity を記録したデータ。 | [16](modules/16_bottleneck_lab_and_profiler.md) |
| self time | 子呼び出しを除いたその operator 自身の時間。 | [16](modules/16_bottleneck_lab_and_profiler.md) |
| torch.compile | PyTorch program を graph 化し、backend compiler で最適化する API。 | [17](modules/17_torch_compile_and_custom_triton.md) |
| graph capture | Python 実行を演算 graph として捕捉すること。 | [17](modules/17_torch_compile_and_custom_triton.md) |
| Inductor | PyTorch 2 系の compiler backend。Triton kernel を生成する場合がある。 | [17](modules/17_torch_compile_and_custom_triton.md) |
| custom Triton kernel | ユーザーが直接書いた Triton kernel。 | [17](modules/17_torch_compile_and_custom_triton.md) |
| fusion boundary | compiler が演算をまたいで融合できなくなる境界。 | [17](modules/17_torch_compile_and_custom_triton.md) |
| fallback | compiler や custom kernel が使えない場合に PyTorch 実装へ戻す経路。 | [17](modules/17_torch_compile_and_custom_triton.md) |
| TTIR | Triton Tensor IR。Triton の高水準中間表現。 | [18](modules/18_ir_ptx_sass_inspection.md) |
| TTGIR | Triton GPU IR。GPU 実行に近づいた中間表現。 | [18](modules/18_ir_ptx_sass_inspection.md) |
| LLVM IR | LLVM compiler infrastructure の中間表現。 | [18](modules/18_ir_ptx_sass_inspection.md) |
| PTX | NVIDIA GPU 向けの仮想 ISA。 | [18](modules/18_ir_ptx_sass_inspection.md) |
| SASS | NVIDIA GPU の実機械語に近い assembly。 | [18](modules/18_ir_ptx_sass_inspection.md) |
| register spill | register に収まらない値が local memory へ退避されること。 | [18](modules/18_ir_ptx_sass_inspection.md) |
| assembly token | `mma` や `ldmatrix` など生成コード上の命令断片。 | [18](modules/18_ir_ptx_sass_inspection.md) |
| stream | GPU 上の操作列。異なる stream は条件が揃えば並行実行できる。 | [19](modules/19_streams_transfers_overlap.md) |
| default stream | 明示指定しない GPU 操作が入る既定の stream。 | [19](modules/19_streams_transfers_overlap.md) |
| pinned memory | page lock された host memory。GPU との非同期転送に有利。 | [19](modules/19_streams_transfers_overlap.md) |
| async copy | CPU が完了を待たずに enqueue する転送。 | [19](modules/19_streams_transfers_overlap.md) |
| overlap | copy と compute などを時間的に重ねること。 | [19](modules/19_streams_transfers_overlap.md) |
| event | GPU stream 上の時刻・同期点を表す object。 | [19](modules/19_streams_transfers_overlap.md) |
| caching allocator | PyTorch が GPU memory を再利用するために保持する allocator。 | [20](modules/20_allocator_zero_fill_lifetime.md) |
| allocated memory | Tensor が現在使用している memory 量。 | [20](modules/20_allocator_zero_fill_lifetime.md) |
| reserved memory | allocator が再利用のために確保したまま保持している memory 量。 | [20](modules/20_allocator_zero_fill_lifetime.md) |
| zero fill | memory 領域を 0 で初期化する write 操作。 | [20](modules/20_allocator_zero_fill_lifetime.md) |
| empty_cache | 未使用 cache block を解放し、他 process から見える空き容量を増やす API。 | [20](modules/20_allocator_zero_fill_lifetime.md) |
| fragmentation | 空き memory が細切れになり、大きな連続領域を取りにくくなる状態。 | [20](modules/20_allocator_zero_fill_lifetime.md) |
| tensor lifetime | Tensor が作られてから参照されなくなるまでの期間。 | [20](modules/20_allocator_zero_fill_lifetime.md) |
| Query/Key/Value | attention の入力 tensor。Query は問い合わせ、Key は照合対象、Value は集約される情報を表す。 | [21](modules/21_attention_pytorch_baselines.md) |
| logits | softmax 前の score。attention では QK^T / sqrt(d)。 | [21](modules/21_attention_pytorch_baselines.md) |
| causal mask | 未来 token を参照しないように score を -inf にする mask。 | [21](modules/21_attention_pytorch_baselines.md) |
| softmax probability | logits を正規化した attention weight。 | [21](modules/21_attention_pytorch_baselines.md) |
| SDPA | scaled dot-product attention。PyTorch では最適化 backend を選ぶ API も含む。 | [21](modules/21_attention_pytorch_baselines.md) |
| materialization | 中間結果を実際の tensor として memory に書き出すこと。 | [21](modules/21_attention_pytorch_baselines.md) |
| IO complexity | 演算ではなく memory read/write の量を中心に見た計算量。 | [22](modules/22_flash_attention_io_accounting.md) |
| online softmax | 全列を一度に保持せず、block ごとに max と分母を更新する softmax。 | [22](modules/22_flash_attention_io_accounting.md) |
| m statistic | softmax の数値安定化に使う行ごとの現在最大値。 | [22](modules/22_flash_attention_io_accounting.md) |
| l statistic | 再スケール済み softmax 分母。 | [22](modules/22_flash_attention_io_accounting.md) |
| output accumulator | 正規化前の weighted value sum を保持する一時値。 | [22](modules/22_flash_attention_io_accounting.md) |
| exact attention | 近似ではなく naive softmax attention と同じ数学的結果を目指す attention。 | [22](modules/22_flash_attention_io_accounting.md) |
| BLOCK_M | 1 program が担当する query 行数。 | [23](modules/23_flash_attention_forward_triton.md) |
| BLOCK_N | 1 回に読む key/value 列数。 | [23](modules/23_flash_attention_forward_triton.md) |
| BLOCK_D | head dimension 方向の block size。 | [23](modules/23_flash_attention_forward_triton.md) |
| causal boundary block | causal mask で full skip でも full valid でもなく、一部だけ有効な block。 | [23](modules/23_flash_attention_forward_triton.md) |
| Q/K/V tile | HBM から読み込む Query/Key/Value の小ブロック。 | [23](modules/23_flash_attention_forward_triton.md) |
| normalization | 最後に accumulator を softmax 分母で割って output にする処理。 | [23](modules/23_flash_attention_forward_triton.md) |
| autotune | 複数 config を実測し、対象 shape/GPU で速いものを選ぶ仕組み。 | [24](modules/24_flash_attention_autotune_portability.md) |
| config | BLOCK_M/N、num_warps、num_stages など kernel の compile-time parameter の組。 | [24](modules/24_flash_attention_autotune_portability.md) |
| meta-parameter | Triton kernel compile 時に固定される parameter。 | [24](modules/24_flash_attention_autotune_portability.md) |
| portability | 複数 GPU/backend で正しく動き、妥当な性能を出せる性質。 | [24](modules/24_flash_attention_autotune_portability.md) |
| architecture-specific tuning | 特定 GPU 世代・backend に合わせた tuning。 | [24](modules/24_flash_attention_autotune_portability.md) |
| validation matrix | dtype、shape、mask 条件などを組み合わせた検証ケース一覧。 | [25](modules/25_flash_attention_validation_matrix.md) |
| pass/fail/skip | 検証結果の状態。unsupported case を fail ではなく skip として管理する。 | [25](modules/25_flash_attention_validation_matrix.md) |
| shape sweep | seq/head_dim/batch/head 数などを変えながら検証すること。 | [25](modules/25_flash_attention_validation_matrix.md) |
| dtype sweep | fp32/fp16/bf16 など dtype を変えながら検証すること。 | [25](modules/25_flash_attention_validation_matrix.md) |
| determinism | 同じ入力で同じ出力が再現される性質。 | [25](modules/25_flash_attention_validation_matrix.md) |
| gradient | loss に対する tensor の微分。 | [26](modules/26_flash_attention_backward_plan.md) |
| dQ/dK/dV | attention backward で求める Query/Key/Value の gradient。 | [26](modules/26_flash_attention_backward_plan.md) |
| recomputation | memory 節約のため forward 中間値を保存せず backward で再計算する方針。 | [26](modules/26_flash_attention_backward_plan.md) |
| saved tensor | backward のため forward で保存する tensor。 | [26](modules/26_flash_attention_backward_plan.md) |
| dropout mask | dropout で残した要素を再現するための mask。 | [26](modules/26_flash_attention_backward_plan.md) |
| causal gradient | causal mask を考慮した backward の gradient。 | [26](modules/26_flash_attention_backward_plan.md) |
| Nsight Compute | NVIDIA CUDA kernel の詳細 metric を見る profiler。 | [27](modules/27_ncu_rocprof_workflow.md) |
| Nsight Systems | CPU/GPU/stream の時間軸を俯瞰する profiler。 | [27](modules/27_ncu_rocprof_workflow.md) |
| rocprof | AMD ROCm 環境で GPU kernel metric を収集する profiler。 | [27](modules/27_ncu_rocprof_workflow.md) |
| metric | bandwidth、occupancy、stall など profiler が計測する指標。 | [27](modules/27_ncu_rocprof_workflow.md) |
| tensor pipe utilization | Tensor Core/MFMA 系演算器の使用率に近い metric。 | [27](modules/27_ncu_rocprof_workflow.md) |
| stall reason | warp が実行できず待っている理由。 | [27](modules/27_ncu_rocprof_workflow.md) |
| API contract | kernel が受け付ける dtype、shape、layout、mask、error 処理の仕様。 | [28](modules/28_production_kernel_checklist.md) |
| CI matrix | GPU/backend/PyTorch/Triton/dtype/shape の組み合わせを自動テストする表。 | [28](modules/28_production_kernel_checklist.md) |
| fallback path | unsupported case で安全な PyTorch/SDPA 実装へ戻る経路。 | [28](modules/28_production_kernel_checklist.md) |
| variable length | batch 内で sequence length が異なるケース。 | [28](modules/28_production_kernel_checklist.md) |
| paged KV cache | LLM 推論で KV cache を page 単位で管理する方式。 | [28](modules/28_production_kernel_checklist.md) |
| GQA/MQA | Grouped Query Attention / Multi-Query Attention。Q head と KV head の数が異なる attention variant。 | [28](modules/28_production_kernel_checklist.md) |
| version pinning | compiler/runtime の version を固定し、再現性を確保すること。 | [28](modules/28_production_kernel_checklist.md) |

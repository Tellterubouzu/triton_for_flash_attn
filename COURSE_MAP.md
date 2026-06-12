# Course Map

この教材の主経路は **docs -> exercises -> lessons** です。

- `docs/modules/NN_*.md`: その章の用語、数式、背景を読む。
- `exercises/NN_*.md`: lesson を見ずに課題を解く。
- `lessons/NN_*.py`: exercise の標準解答・確認用 script として実行する。

旧来のトピック別メモは `docs/reference/` に移動しました。主経路から必要なものだけリンクしています。

## 全対応表

| Stage | Docs | Exercise | Lesson answer | 新出用語の例 |
|---:|---|---|---|---|
| 00 | [`docs/modules/00_setup_and_sanity.md`](docs/modules/00_setup_and_sanity.md) | [`exercises/00_setup_and_sanity.md`](exercises/00_setup_and_sanity.md) | [`lessons/00_setup_and_sanity.py`](lessons/00_setup_and_sanity.py) | GPU, CUDA, ROCm |
| 01 | [`docs/modules/01_gpu_execution_model.md`](docs/modules/01_gpu_execution_model.md) | [`exercises/01_gpu_execution_model.md`](exercises/01_gpu_execution_model.md) | [`lessons/01_gpu_execution_model.py`](lessons/01_gpu_execution_model.py) | launch, grid, thread block |
| 02 | [`docs/modules/02_warps_sms_occupancy.md`](docs/modules/02_warps_sms_occupancy.md) | [`exercises/02_warps_sms_occupancy.md`](exercises/02_warps_sms_occupancy.md) | [`lessons/02_warps_sms_occupancy.py`](lessons/02_warps_sms_occupancy.py) | SM, occupancy, resident program |
| 03 | [`docs/modules/03_memory_hierarchy_and_device_query.md`](docs/modules/03_memory_hierarchy_and_device_query.md) | [`exercises/03_memory_hierarchy_and_device_query.md`](exercises/03_memory_hierarchy_and_device_query.md) | [`lessons/03_memory_hierarchy_and_device_query.py`](lessons/03_memory_hierarchy_and_device_query.py) | HBM, L2 cache, SRAM |
| 04 | [`docs/modules/04_tensor_addresses_strides_layouts.md`](docs/modules/04_tensor_addresses_strides_layouts.md) | [`exercises/04_tensor_addresses_strides_layouts.md`](exercises/04_tensor_addresses_strides_layouts.md) | [`lessons/04_tensor_addresses_strides_layouts.py`](lessons/04_tensor_addresses_strides_layouts.py) | pointer, byte address, data_ptr |
| 05 | [`docs/modules/05_hbm_bandwidth_and_copy.md`](docs/modules/05_hbm_bandwidth_and_copy.md) | [`exercises/05_hbm_bandwidth_and_copy.md`](exercises/05_hbm_bandwidth_and_copy.md) | [`lessons/05_hbm_bandwidth_and_copy.py`](lessons/05_hbm_bandwidth_and_copy.py) | effective bandwidth, read traffic, write traffic |
| 06 | [`docs/modules/06_coalescing_strides_cache.md`](docs/modules/06_coalescing_strides_cache.md) | [`exercises/06_coalescing_strides_cache.md`](exercises/06_coalescing_strides_cache.md) | [`lessons/06_coalescing_strides_cache.py`](lessons/06_coalescing_strides_cache.py) | coalescing, memory transaction, cache line/sector |
| 07 | [`docs/modules/07_vector_add_first_kernel.md`](docs/modules/07_vector_add_first_kernel.md) | [`exercises/07_vector_add_first_kernel.md`](exercises/07_vector_add_first_kernel.md) | [`lessons/07_vector_add_first_kernel.py`](lessons/07_vector_add_first_kernel.py) | tl.program_id, tl.arange, tl.load |
| 08 | [`docs/modules/08_elementwise_fusion.md`](docs/modules/08_elementwise_fusion.md) | [`exercises/08_elementwise_fusion.md`](exercises/08_elementwise_fusion.md) | [`lessons/08_elementwise_fusion.py`](lessons/08_elementwise_fusion.py) | fusion, intermediate tensor, arithmetic intensity |
| 09 | [`docs/modules/09_cache_hints_and_eviction.md`](docs/modules/09_cache_hints_and_eviction.md) | [`exercises/09_cache_hints_and_eviction.md`](exercises/09_cache_hints_and_eviction.md) | [`lessons/09_cache_hints_and_eviction.py`](lessons/09_cache_hints_and_eviction.py) | cache modifier, eviction policy, .ca |
| 10 | [`docs/modules/10_block_pointers_boundary_check.md`](docs/modules/10_block_pointers_boundary_check.md) | [`exercises/10_block_pointers_boundary_check.md`](exercises/10_block_pointers_boundary_check.md) | [`lessons/10_block_pointers_boundary_check.py`](lessons/10_block_pointers_boundary_check.py) | block pointer, tl.make_block_ptr, boundary_check |
| 11 | [`docs/modules/11_reductions_and_softmax.md`](docs/modules/11_reductions_and_softmax.md) | [`exercises/11_reductions_and_softmax.md`](exercises/11_reductions_and_softmax.md) | [`lessons/11_reductions_and_softmax.py`](lessons/11_reductions_and_softmax.py) | reduction, row-wise, stable softmax |
| 12 | [`docs/modules/12_numerics_and_correctness.md`](docs/modules/12_numerics_and_correctness.md) | [`exercises/12_numerics_and_correctness.md`](exercises/12_numerics_and_correctness.md) | [`lessons/12_numerics_and_correctness.py`](lessons/12_numerics_and_correctness.py) | fp32, fp16, bf16 |
| 13 | [`docs/modules/13_tiled_matmul.md`](docs/modules/13_tiled_matmul.md) | [`exercises/13_tiled_matmul.md`](exercises/13_tiled_matmul.md) | [`lessons/13_tiled_matmul.py`](lessons/13_tiled_matmul.py) | GEMM, tile, K loop |
| 14 | [`docs/modules/14_tensor_cores_and_tl_dot.md`](docs/modules/14_tensor_cores_and_tl_dot.md) | [`exercises/14_tensor_cores_and_tl_dot.md`](exercises/14_tensor_cores_and_tl_dot.md) | [`lessons/14_tensor_cores_and_tl_dot.py`](lessons/14_tensor_cores_and_tl_dot.py) | Tensor Core, MMA, MFMA |
| 15 | [`docs/modules/15_layernorm_and_rowwise_norm.md`](docs/modules/15_layernorm_and_rowwise_norm.md) | [`exercises/15_layernorm_and_rowwise_norm.md`](exercises/15_layernorm_and_rowwise_norm.md) | [`lessons/15_layernorm_and_rowwise_norm.py`](lessons/15_layernorm_and_rowwise_norm.py) | LayerNorm, mean, variance |
| 16 | [`docs/modules/16_bottleneck_lab_and_profiler.md`](docs/modules/16_bottleneck_lab_and_profiler.md) | [`exercises/16_bottleneck_lab_and_profiler.md`](exercises/16_bottleneck_lab_and_profiler.md) | [`lessons/16_bottleneck_lab_and_profiler.py`](lessons/16_bottleneck_lab_and_profiler.py) | profiler, hotspot, operator |
| 17 | [`docs/modules/17_torch_compile_and_custom_triton.md`](docs/modules/17_torch_compile_and_custom_triton.md) | [`exercises/17_torch_compile_and_custom_triton.md`](exercises/17_torch_compile_and_custom_triton.md) | [`lessons/17_torch_compile_and_custom_triton.py`](lessons/17_torch_compile_and_custom_triton.py) | torch.compile, graph capture, Inductor |
| 18 | [`docs/modules/18_ir_ptx_sass_inspection.md`](docs/modules/18_ir_ptx_sass_inspection.md) | [`exercises/18_ir_ptx_sass_inspection.md`](exercises/18_ir_ptx_sass_inspection.md) | [`lessons/18_ir_ptx_sass_inspection.py`](lessons/18_ir_ptx_sass_inspection.py) | TTIR, TTGIR, LLVM IR |
| 19 | [`docs/modules/19_streams_transfers_overlap.md`](docs/modules/19_streams_transfers_overlap.md) | [`exercises/19_streams_transfers_overlap.md`](exercises/19_streams_transfers_overlap.md) | [`lessons/19_streams_transfers_overlap.py`](lessons/19_streams_transfers_overlap.py) | stream, default stream, pinned memory |
| 20 | [`docs/modules/20_allocator_zero_fill_lifetime.md`](docs/modules/20_allocator_zero_fill_lifetime.md) | [`exercises/20_allocator_zero_fill_lifetime.md`](exercises/20_allocator_zero_fill_lifetime.md) | [`lessons/20_allocator_zero_fill_lifetime.py`](lessons/20_allocator_zero_fill_lifetime.py) | caching allocator, allocated memory, reserved memory |
| 21 | [`docs/modules/21_attention_pytorch_baselines.md`](docs/modules/21_attention_pytorch_baselines.md) | [`exercises/21_attention_pytorch_baselines.md`](exercises/21_attention_pytorch_baselines.md) | [`lessons/21_attention_pytorch_baselines.py`](lessons/21_attention_pytorch_baselines.py) | Query/Key/Value, logits, causal mask |
| 22 | [`docs/modules/22_flash_attention_io_accounting.md`](docs/modules/22_flash_attention_io_accounting.md) | [`exercises/22_flash_attention_io_accounting.md`](exercises/22_flash_attention_io_accounting.md) | [`lessons/22_flash_attention_io_accounting.py`](lessons/22_flash_attention_io_accounting.py) | IO complexity, online softmax, m statistic |
| 23 | [`docs/modules/23_flash_attention_forward_triton.md`](docs/modules/23_flash_attention_forward_triton.md) | [`exercises/23_flash_attention_forward_triton.md`](exercises/23_flash_attention_forward_triton.md) | [`lessons/23_flash_attention_forward_triton.py`](lessons/23_flash_attention_forward_triton.py) | BLOCK_M, BLOCK_N, BLOCK_D |
| 24 | [`docs/modules/24_flash_attention_autotune_portability.md`](docs/modules/24_flash_attention_autotune_portability.md) | [`exercises/24_flash_attention_autotune_portability.md`](exercises/24_flash_attention_autotune_portability.md) | [`lessons/24_flash_attention_autotune_portability.py`](lessons/24_flash_attention_autotune_portability.py) | autotune, config, meta-parameter |
| 25 | [`docs/modules/25_flash_attention_validation_matrix.md`](docs/modules/25_flash_attention_validation_matrix.md) | [`exercises/25_flash_attention_validation_matrix.md`](exercises/25_flash_attention_validation_matrix.md) | [`lessons/25_flash_attention_validation_matrix.py`](lessons/25_flash_attention_validation_matrix.py) | validation matrix, pass/fail/skip, shape sweep |
| 26 | [`docs/modules/26_flash_attention_backward_plan.md`](docs/modules/26_flash_attention_backward_plan.md) | [`exercises/26_flash_attention_backward_plan.md`](exercises/26_flash_attention_backward_plan.md) | [`lessons/26_flash_attention_backward_plan.py`](lessons/26_flash_attention_backward_plan.py) | gradient, dQ/dK/dV, recomputation |
| 27 | [`docs/modules/27_ncu_rocprof_workflow.md`](docs/modules/27_ncu_rocprof_workflow.md) | [`exercises/27_ncu_rocprof_workflow.md`](exercises/27_ncu_rocprof_workflow.md) | [`lessons/27_ncu_rocprof_workflow.py`](lessons/27_ncu_rocprof_workflow.py) | Nsight Compute, Nsight Systems, rocprof |
| 28 | [`docs/modules/28_production_kernel_checklist.md`](docs/modules/28_production_kernel_checklist.md) | [`exercises/28_production_kernel_checklist.md`](exercises/28_production_kernel_checklist.md) | [`lessons/28_production_kernel_checklist.py`](lessons/28_production_kernel_checklist.py) | API contract, CI matrix, fallback path |

## 検査

未対応 exercise や link 漏れは次で確認できます。

```bash
python scripts/validate_course_graph.py
```

## 推奨の進め方

1. `docs/modules/00_setup_and_sanity.md` から順に読む。
2. 同じ番号の `exercises/` を解く。
3. 同じ番号の `lessons/` を実行して、自分の理解と実装を比較する。
4. 必要になったときだけ `docs/reference/` を読む。

# 10. Cache hints, IR, PTX/ISA inspection, and profiling

この章の目的は、Triton source と実際に backend が生成する code の距離を理解することです。

## cache_modifier

Triton の `tl.load` は NVIDIA PTX 向けに `cache_modifier` を受け取ります。代表値は次です。

- `".ca"`: all levels に cache する。
- `".cg"`: global level、つまり L2 以下に cache し、L1 を避ける方向の hint。
- `".cv"`: cache せず再取得する方向の hint。

`tl.store` にも `".wb"`, `".cg"`, `".cs"`, `".wt"` などの cache modifier があります。これらは hint であり、backend と architecture に依存します。NVIDIA 以外では意味が変わる、無視される、または compiler version に依存する可能性があります。

## eviction_policy

`evict_first` は再利用の少ない streaming access、`evict_last` は再利用したい tile に対する hint として実験します。FlashAttention では Q block や accumulator は再利用が高く、K/V は block streaming されます。この access pattern に cache hint が効くかどうかは GPU と shape に依存するため、固定観念ではなく benchmark で判断します。

## IR/assembly の保存

Triton kernel call object から `asm` を読める環境では、TTIR、TTGIR、LLVM IR、PTX などを保存できます。

```python
compiled = kernel[grid](...)
print(compiled.asm.keys())
print(compiled.asm.get("ptx", ""))
```

Triton と backend の version により key は変わります。教材の `lessons/18_ir_ptx_sass_inspection.py` は存在する key だけ保存します。

## profiler counters

PyTorch profiler は operator table と memory allocation の入口として便利ですが、L2 hit rate、warp stall、shared memory bank conflict などの低レイヤー counter は Nsight Compute / rocprof を使います。教材ではまず PyTorch profiler で candidate kernel を特定し、その後 vendor profiler に渡す形にします。

## 実験

```bash
python lessons/09_cache_hints_and_eviction.py --numel 67108864 --dtype fp16
python lessons/18_ir_ptx_sass_inspection.py --out-dir ir_dumps
```

# 09. Transfer, zeroing, and PyTorch allocator behavior

この章の目的は、計算 kernel の外側にある bottleneck、つまり host-device transfer、device-device copy、zero fill、allocator cache を測ることです。

## Host to device / device to host

CPU Tensor から GPU Tensor への転送は、PCIe または NVLink/UMA などの接続に依存します。PyTorch では pinned host memory と `non_blocking=True` を組み合わせると、非同期転送を使える条件が揃います。

```python
x_cpu = torch.empty((1 << 28,), dtype=torch.float16, pin_memory=True)
x_gpu = x_cpu.to("cuda", non_blocking=True)
```

ただし、非同期転送を正しく測るには stream と synchronization を明示する必要があります。

## Device to device copy

GPU 内の copy は HBM read + HBM write です。copy bandwidth benchmark は、その GPU で期待できる上限の一部を測る簡易実験になります。

```bash
python benchmarks/bench_memory_bandwidth.py --numel 268435456 --dtype fp16
```

## zero fill は無料ではない

次の式は、zero fill の HBM traffic の最小量を表します。

\[
\mathrm{bytes} = N \cdot \mathrm{element\_size}
\]

zero は write だけに見えますが、実際の memory subsystem では write allocate や cache policy の影響があります。Triton の zero kernel、`torch.zeros_like`、`x.zero_()` は別の kernel/allocator path を通るため、実測で比較します。

## allocator の見方

PyTorch の CUDA allocator では、`memory_allocated` は Tensor が実際に占有している memory、`memory_reserved` は allocator が driver から確保して保持している memory を見る目安です。`del x` 後も reserved が残ることがあります。これは次回 allocation の高速化のためです。

```python
from triton_flash_course.hardware import allocator_snapshot
print(allocator_snapshot())
```

## FlashAttention との関係

FlashAttention kernel 本体が速くても、Q/K/V layout 変換、contiguous 化、dtype cast、KV cache copy、zero fill が周辺にあると end-to-end では遅くなります。`profile_attention.py` だけでなく transfer/allocator lesson も併用して、モデル全体の memory movement を見る必要があります。

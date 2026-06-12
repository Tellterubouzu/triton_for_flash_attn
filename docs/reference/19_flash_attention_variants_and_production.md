# FlashAttention Variants and Production Work

## forward-only から実運用への差分

この教材の kernel は forward-only です。実運用では、次の feature が必要になることがあります。

## backward

Backward では、\(dO\) から \(dQ, dK, dV\) を計算します。Forward と同様に score/probability を materialize しない方針を取る場合、forward の softmax 統計量、再計算、tile 分割、atomic accumulation の扱いを設計する必要があります。

## dropout

Training attention では dropout mask の再現性が問題になります。forward で mask を保存するのか、RNG seed/offset から backward で再生成するのかを決めます。

## variable length / packed sequence

実データでは sequence length が batch 内で異なります。padding で揃えると無駄な計算が増えるため、prefix sum / cu_seqlens 型の metadata を使った packed layout が必要になります。

## GQA / MQA

Grouped Query Attention や Multi-Query Attention では、Q heads と KV heads の対応が 1:1 ではありません。addressing と program grid を変える必要があります。

## KV cache / paged attention

Decoding では query length が短く、KV cache が長くなります。training 用 FlashAttention と bottleneck が変わり、page table、cache layout、small-batch launch overhead が重要になります。

## bias / ALiBi / sliding window

Score に bias を足す variant では、score tile を作った直後、softmax 前に bias を融合します。ただし bias の layout が悪いと、せっかく削減した HBM traffic が戻ります。

## production fallback

Unsupported dtype/shape/device では、明示的に `torch.nn.functional.scaled_dot_product_attention` へ fallback します。無言で遅い path に落ちる設計は避けます。

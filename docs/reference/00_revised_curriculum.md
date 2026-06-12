# Revised Curriculum Rationale

## 変更前の問題

初期版は Triton 文法から入り、その後に memory hierarchy、address、warp、Tensor Core を追加していました。この順序だと、`tl.load` や `tl.dot` を「動く API」としては使えても、なぜその tile size が速いのか、なぜ stride が遅いのか、なぜ FlashAttention が HBM temporary を消せるのかを説明しにくくなります。

## 変更後の順序

新しい順序は次の依存関係に合わせています。

```text
GPU execution model
  -> memory hierarchy and address
  -> Triton pointer/tile primitives
  -> reduction and dot primitives
  -> profiler/compiler/IR
  -> attention IO schedule
  -> FlashAttention kernel
  -> validation/autotune/production
```

## 判断基準

Triton の lesson は、次のいずれかを満たす場合だけ前半に置きます。

1. 後続のすべての kernel で使う mental model である。
2. correctness に影響する。
3. performance tuning の観測方法に影響する。
4. FlashAttention の IO schedule に直接必要である。

このため、kernel / warp / SM / HBM / address / stride は最初に移動しました。一方、external profiler、production checklist、backward design は、forward kernel を一度書いてからの方が具体的に理解しやすいため最後に置いています。

## 学習時のルール

- performance tuning より前に correctness matrix を作る。
- profiler counter を見る前に、bytes と FLOPs の下限を概算する。
- `torch.compile` で十分な fusion は manual Triton にしない。
- single GPU で速い config を portable default だとみなさない。
- 生成コード token 検索は補助情報であり、最終判断は profiler metric で行う。

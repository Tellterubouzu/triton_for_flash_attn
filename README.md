# Triton FlashAttention Course

PyTorch 実装を profile して bottleneck を特定し、必要な箇所だけ Triton kernel に置き換えるための教材です。最終目標は、NVIDIA / AMD の複数 GPU で検証・チューニングできる FlashAttention forward kernel を自力で設計、実装、検証、改善できる状態です。

## 学習順序

この版では、教材の主経路を **docs -> exercises -> lessons** に統一しています。

```text
docs/modules/NN_*.md    # 新出用語、数式、低レイヤー背景を読む
  -> exercises/NN_*.md  # lesson を見ずに課題を解く
  -> lessons/NN_*.py    # 答え合わせ・確認用 script を実行する
```

対応表は [`COURSE_MAP.md`](COURSE_MAP.md) に全件あります。用語集は [`docs/glossary.md`](docs/glossary.md) です。

## 対象者

- PyTorch で Transformer / LLM 系の実装を読める。
- CUDA C++ は必須ではないが、kernel、warp、SM、HBM、SRAM/shared memory、Tensor Core などを実装設計に結びつけたい。
- 研究実装や推論最適化で custom kernel が必要になる可能性がある。

## 環境

Triton kernel の実行・benchmark には GPU が必要です。CPU でも一部 lesson は読めます。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,profile]"
```

GPU と Triton が有効か確認します。

```bash
python lessons/00_setup_and_sanity.py
```

## 最初に読むファイル

```bash
python scripts/print_course_sequence.py
python scripts/validate_course_graph.py
```

その後、次の順に進めます。

1. [`docs/modules/00_setup_and_sanity.md`](docs/modules/00_setup_and_sanity.md)
2. [`exercises/00_setup_and_sanity.md`](exercises/00_setup_and_sanity.md)
3. [`lessons/00_setup_and_sanity.py`](lessons/00_setup_and_sanity.py)

以降は番号を 01, 02, ... と進めます。

## リポジトリ構成

```text
src/triton_flash_course/   # PyTorch baseline、Triton kernel、測定 helper
docs/modules/              # 主教材。新出用語と数式を説明し、exercise へ誘導する
docs/reference/            # 深掘り用の補足資料
docs/glossary.md           # 用語集
exercises/                 # docs に対応する課題。lessons が答えになる
lessons/                   # 課題の標準解答・確認用 script
benchmarks/                # kernel ごとの benchmark / profiler
tests/                     # correctness tests
scripts/                   # course graph 検査、smoke test、順序表示
```

## 実行例

```bash
# 教材対応関係の検査
python scripts/validate_course_graph.py

# GPU がなくても読める smoke check。Triton 実行は skip されます。
python scripts/run_all_smoke_tests.py

# GPU がある環境で correctness test
pytest -m gpu

# attention の benchmark
python benchmarks/bench_attention.py --batch 2 --heads 8 --seq 1024 --dim 64 --dtype fp16

# PyTorch profiler trace を作る
python benchmarks/profile_attention.py --trace-dir traces --seq 2048 --dim 64
```

## この教材で扱う FlashAttention

この数式は、Query と Key の類似度を softmax で正規化し、Value の加重和を取る scaled dot-product attention を表します。

\[
O = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

naive 実装は \(QK^\top\) と softmax probability \(P\) を明示的に HBM に materialize します。FlashAttention は Query block と Key/Value block に分割し、各 Query block ごとに online softmax の統計量 \(m_i, l_i\) と output accumulator を SRAM/register 側で更新します。これにより exact attention のまま、中間行列 \(P\) を HBM に書かずに計算します。

このリポジトリの FlashAttention kernel は学習用 forward-only です。実運用向けの dropout、variable length、paged KV cache、GQA/MQA、FP8、persistent scheduling、split-K、deterministic backward は未実装です。後半の exercise と checklist で、それらを設計課題として扱います。

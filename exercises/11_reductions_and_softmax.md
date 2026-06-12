# Exercise 11: row-wise reduction と stable softmax

## 対応する docs

先に [`docs/modules/11_reductions_and_softmax.md`](../docs/modules/11_reductions_and_softmax.md) を読んでください。新しい用語は docs の「新出用語」で定義しています。

## 課題

1. [`docs/modules/11_reductions_and_softmax.md`](../docs/modules/11_reductions_and_softmax.md) を読み、新出用語のうち **reduction, row-wise, stable softmax, max trick, power-of-two block** を 1 文ずつ自分の言葉で説明してください。
2. lesson を開く前に、この module で何を測る・実装する・観察するべきかを 3 点に分けて書いてください。
3. PyTorch baseline と Triton kernel、または CPU/GPU 観察結果を比較するとき、どの指標を見ればよいかを決めてください。
4. 予想される bottleneck を memory-bound / compute-bound / launch-bound / correctness-risk のいずれかに分類してください。
5. stable softmax の max trick が結果を変えず overflow を避ける理由を説明してください。

## 提出メモの形式

```text
用語メモ:
- ...

予想:
- bottleneck: ...
- correctness risk: ...

観察結果:
- ...

次に変更するなら:
- ...
```

## 解答・確認用 lesson

演習を解いた後で [`lessons/11_reductions_and_softmax.py`](../lessons/11_reductions_and_softmax.py) を実行してください。この script がこの exercise の標準解答・確認用実装です。

```bash
python lessons/11_reductions_and_softmax.py
```

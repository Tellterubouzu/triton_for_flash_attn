# Numerics and Correctness Policy

## この章の目的

Triton kernel の correctness は「PyTorch と完全に同じ bit pattern か」ではなく、「同じ数式を、許容される dtype 誤差内で計算しているか」で判断します。FlashAttention は softmax probability を HBM に materialize せず、online softmax で accumulator を更新するため、naive PyTorch と演算順序が異なります。

## stable softmax

この数式は、logits の最大値を引いて softmax の overflow を避ける変形を表します。

\[
y_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}, \quad m = \max_j x_j
\]

softmax は全要素に同じ定数を足し引きしても変わらないため、最大値を引いても数学的な値は同じです。ただし有限精度では、`exp(1000)` のような overflow を避ける効果があります。

## FlashAttention の online softmax

この数式は、既に処理した key block の softmax 統計量を、新しい score block を読んだ後の最大値へ再スケールする更新式です。

\[
m_{new}=\max(m_{old}, \max S_{block})
\]

\[
l_{new}=e^{m_{old}-m_{new}}l_{old}+\sum_j e^{S_j-m_{new}}
\]

\[
acc_{new}=e^{m_{old}-m_{new}}acc_{old}+\sum_j e^{S_j-m_{new}}V_j
\]

ここで \(m\) は row-wise max、\(l\) は softmax denominator、\(acc\) は未正規化の output accumulator です。最後に \(O=acc/l\) とします。

## tolerance の考え方

- fp32 elementwise: 厳しめの `rtol=1e-4`, `atol=1e-5` から始める。
- fp16 reduction/attention: `rtol=3e-2`, `atol=3e-2` 程度から始め、shape ごとに実測する。
- bf16 reduction/attention: bf16 は mantissa が短いため fp16 より緩い tolerance が必要になることがあります。
- backward は forward より誤差が増えるため、gradient check と実モデル loss の両方で見る。

教材の `src/triton_flash_course/numerics.py` は、error report と推奨 tolerance を返します。

```bash
python lessons/12_numerics_and_correctness.py --seq 128 --dim 64 --dtype fp16
```

## correctness matrix

FlashAttention kernel は、最低限次の軸で検証します。

- dtype: fp16, bf16, 必要なら fp32/tf32 policy
- sequence length: 1, small prime, power-of-two, non-power-of-two, large
- head_dim: 16, 32, 64, 128
- causal: true / false
- layout: contiguous と view/stride をどう扱うか
- special values: NaN/Inf を許すか、入力で弾くか

```bash
python lessons/25_flash_attention_validation_matrix.py --max-cases 32
```

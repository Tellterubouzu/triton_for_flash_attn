"""Lesson 26: FlashAttention backward design plan.

この lesson は実装ではなく設計演習です。FlashAttention forward で保存した log-sum-exp と出力 O を使い、
backward をどのように block 化するかを考えます。

主な式:

    P = softmax(S), S = QK^T / sqrt(d)
    dV = P^T dO
    dP = dO V^T
    dS = P * (dP - sum_j P_j dP_j)
    dQ = dS K / sqrt(d)
    dK = dS^T Q / sqrt(d)

課題は exercises/26_flash_attention_backward_plan.md を参照してください。
"""
from __future__ import annotations


def main() -> None:
    print(__doc__)
    print("この段階では forward kernel の correctness/profiling/autotune が安定していることを前提に、backward の分割戦略を設計します。")


if __name__ == "__main__":
    main()

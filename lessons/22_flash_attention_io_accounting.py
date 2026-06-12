from __future__ import annotations

import argparse
import math

from _bootstrap import add_src_to_path

add_src_to_path()


def gib(x: int | float) -> float:
    return float(x) / 1024**3


def attention_io(batch: int, heads: int, seq: int, dim: int, bytes_per_elem: int, block_m: int, save_lse: bool) -> dict[str, int]:
    qkv = 3 * batch * heads * seq * dim * bytes_per_elem
    out = batch * heads * seq * dim * bytes_per_elem
    scores = batch * heads * seq * seq * bytes_per_elem
    probs = batch * heads * seq * seq * bytes_per_elem
    lse = batch * heads * seq * 4 if save_lse else 0

    # Very rough lower-bound. K and V are streamed once per query block.
    num_q_blocks = math.ceil(seq / block_m)
    flash_reads = batch * heads * (
        seq * dim * bytes_per_elem +  # Q once
        num_q_blocks * 2 * seq * dim * bytes_per_elem  # K and V per Q block
    )
    flash_writes = out + lse
    return {
        "qkv_bytes": qkv,
        "out_bytes": out,
        "naive_score_temp_bytes": scores,
        "naive_prob_temp_bytes": probs,
        "naive_temp_total_bytes": scores + probs,
        "flash_lse_bytes": lse,
        "flash_lower_bound_traffic_bytes": flash_reads + flash_writes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--bytes-per-elem", type=int, default=2)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--save-lse", action="store_true")
    args = parser.parse_args()

    d = attention_io(args.batch, args.heads, args.seq, args.dim, args.bytes_per_elem, args.block_m, args.save_lse)
    print(f"Shape: B={args.batch}, H={args.heads}, N={args.seq}, D={args.dim}, elem={args.bytes_per_elem} bytes")
    for k, v in d.items():
        print(f"{k:36s}: {gib(v):10.3f} GiB")

    print("\nNaive attention materializes score/probability [B,H,N,N].")
    print("FlashAttention avoids those HBM temporaries, but repeatedly streams K/V for each query block.")
    print("The estimate above is a schedule-level lower bound, not a profiler counter.")


if __name__ == "__main__":
    main()

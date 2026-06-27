#!/usr/bin/env python3
"""Measure GPU latency for attention-only QuadTree Attention."""

from __future__ import annotations

import argparse
from pathlib import Path

from qt_bench import ITERS, K_VALUES, N_VALUES, RESULTS_DIR, WARMUP, run_latency_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=list(N_VALUES))
    parser.add_argument("--k", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "gpu_latency.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_latency_sweep(
        backend="cuda_ref",
        n_values=args.n,
        k_values=args.k,
        out=args.out,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()

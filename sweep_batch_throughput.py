#!/usr/bin/env python3
"""Sweep input batch size and report occupancy-normalized GPU throughput.

Work is treated as proportional to batch * heads (fixed n, k, levels, head_dim).
Throughput is then work / measured latency, normalized so H=1 B=1 is 1.0.

Run on the H100 machine, for example:

  python3 sweep_batch_throughput.py --backend triton --n 4096 --k 8
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch

from qt_bench import (
    HEAD_DIM,
    RESULTS_DIR,
    build_case,
    get_device,
    make_workload,
    measure_latency,
    print_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda_ref", "triton"), default="triton")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "gpu_batch_throughput.csv",
    )
    return parser.parse_args()


def power_of_two_range(start: int, stop: int) -> list[int]:
    values = []
    value = start
    while value <= stop:
        values.append(value)
        value *= 2
    return values


def relative_ops(heads: int, batch: int) -> float:
    return float(heads * batch)


def measure_one(
    *,
    backend: str,
    n: int,
    k: int,
    heads: int,
    batch: int,
    head_dim: int,
    warmup: int,
    iters: int,
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    config, attn, queries, keys, values = build_case(
        backend=backend,
        n=n,
        k=k,
        heads=heads,
        head_dim=head_dim,
        batch=batch,
    )
    run_once = make_workload(attn, queries, keys, values, get_device(backend))
    result = measure_latency(
        config=config,
        run_once=run_once,
        warmup=warmup,
        iters=iters,
    )
    del attn, queries, keys, values, run_once
    return result


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sweeps = (
        (1, power_of_two_range(1, 1024)),
        (8, power_of_two_range(1, 256)),
    )

    print(
        f"Sweeping {args.backend} n={args.n} k={args.k} "
        f"warmup={args.warmup} iters={args.iters}"
    )

    baseline_latency_ms = None
    rows: list[dict[str, float | int | str]] = []
    peaks: dict[int, dict[str, float | int]] = {}

    for heads, batches in sweeps:
        for batch in batches:
            try:
                result = measure_one(
                    backend=args.backend,
                    n=args.n,
                    k=args.k,
                    heads=heads,
                    batch=batch,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" not in message:
                    raise
                print(f"OOM heads={heads} batch={batch}; stopping this head sweep.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

            print_result(result)
            latency_ms = result.latency_ms_mean
            ops = relative_ops(heads, batch)
            throughput = ops / latency_ms
            if baseline_latency_ms is None:
                if heads != 1 or batch != 1:
                    raise SystemExit("First measurement must be H=1 B=1 for normalization.")
                baseline_latency_ms = latency_ms
                baseline_throughput = relative_ops(1, 1) / baseline_latency_ms
            throughput_norm = throughput / baseline_throughput

            row = {
                "backend": result.backend,
                "n": result.n,
                "k": result.k,
                "heads": heads,
                "batch": batch,
                "latency_ms_mean": latency_ms,
                "latency_ms_median": result.latency_ms_median,
                "relative_ops": ops,
                "throughput": throughput,
                "throughput_norm": throughput_norm,
            }
            rows.append(row)
            print(
                f"THROUGHPUT heads={heads} batch={batch} "
                f"rel_ops={ops:.0f} tput_norm={throughput_norm:.3f}"
            )

            peak = peaks.get(heads)
            if peak is None or throughput_norm > peak["throughput_norm"]:
                peaks[heads] = {
                    "batch": batch,
                    "latency_ms_mean": latency_ms,
                    "throughput_norm": throughput_norm,
                }

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {args.out}")
    print("Peak throughput (normalized to H=1 B=1):")
    for heads in sorted(peaks):
        peak = peaks[heads]
        print(
            f"  H={heads}: B={peak['batch']}  "
            f"tput_norm={peak['throughput_norm']:.3f}  "
            f"latency_ms={peak['latency_ms_mean']:.4f}"
        )


if __name__ == "__main__":
    main()

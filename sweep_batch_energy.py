#!/usr/bin/env python3
"""Sweep H/B and report occupancy-normalized GPU energy efficiency.

Work is treated as proportional to heads * batch (fixed n, k, levels, head_dim).
Energy per forward is dynamic_power_W * latency_ms (mJ).
Efficiency is work / energy, normalized so H=1 B=1 is 1.0.

Idle power is measured once. Each (H, B) then gets a short latency measurement
and a nvidia-smi workload sample (same window/trim fractions as gpu_power.sh).
Defaults are search-oriented (~5–8 min). For paper-grade watts use
--idle-seconds 60 --power-seconds 45 --warmup 100 --iters 1000.

Run on the H100 machine, for example:

  python3 sweep_batch_energy.py --backend triton --n 4096 --k 8
"""

from __future__ import annotations

import argparse
import csv
import gc
import subprocess
import time
from pathlib import Path

import torch

from qt_bench import (
    HEAD_DIM,
    RESULTS_DIR,
    TRIM,
    build_case,
    get_device,
    make_workload,
    measure_latency,
    print_result,
    run_power_loop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda_ref", "triton"), default="triton")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--idle-seconds", type=float, default=15.0)
    parser.add_argument("--power-seconds", type=float, default=12.0)
    parser.add_argument("--loop-ms", type=int, default=100)
    parser.add_argument("--cooldown-seconds", type=float, default=1.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "gpu_batch_energy.csv",
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


def read_power_samples(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text().splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if not token or token.upper() == "N/A":
            continue
        values.append(float(token.replace(" W", "")))
    if not values:
        raise SystemExit(f"no power samples collected in {path}")
    return values


def summarize_power(
    values: list[float],
    start_fraction: float,
    end_fraction: float,
    trim_fraction: float,
) -> float:
    start = int(len(values) * start_fraction)
    end = int(len(values) * end_fraction)
    start = max(0, min(start, len(values)))
    end = max(start + 1, min(end, len(values)))
    window = values[start:end]
    ordered = sorted(window)
    trim = int(len(ordered) * trim_fraction)
    if trim and len(ordered) > 2 * trim:
        kept = ordered[trim:-trim]
    else:
        kept = ordered
    return sum(kept) / len(kept)


def start_smi(path: Path, loop_ms: int, gpu_index: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={loop_ms}",
        ],
        stdout=handle,
        stderr=subprocess.DEVNULL,
    )
    proc._smi_handle = handle  # type: ignore[attr-defined]
    return proc


def stop_smi(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    handle = getattr(proc, "_smi_handle", None)
    if handle is not None:
        handle.close()


def measure_idle_power(*, seconds: float, loop_ms: int, gpu_index: int, out_dir: Path) -> float:
    path = out_dir / "idle_power_w.txt"
    print(f"[energy-sweep] sampling idle power for {seconds:.0f}s")
    proc = start_smi(path, loop_ms, gpu_index)
    time.sleep(seconds)
    stop_smi(proc)
    return summarize_power(read_power_samples(path), 0.50, 1.00, TRIM)


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
    power_seconds: float,
    loop_ms: int,
    gpu_index: int,
    sample_path: Path,
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
    proc = start_smi(sample_path, loop_ms, gpu_index)
    iterations = run_power_loop(run_once, power_seconds, backend)
    stop_smi(proc)
    workload_w = summarize_power(read_power_samples(sample_path), 0.15, 0.85, TRIM)
    del attn, queries, keys, values, run_once
    return result, workload_w, iterations


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = args.out.parent / f"{args.out.stem}_power_traces"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sweeps = (
        (1, power_of_two_range(1, 1024)),
        (8, power_of_two_range(1, 256)),
    )

    print(
        f"Sweeping {args.backend} n={args.n} k={args.k} "
        f"warmup={args.warmup} iters={args.iters} "
        f"idle={args.idle_seconds:.0f}s work={args.power_seconds:.0f}s"
    )

    idle_w = measure_idle_power(
        seconds=args.idle_seconds,
        loop_ms=args.loop_ms,
        gpu_index=args.gpu_index,
        out_dir=sample_dir,
    )
    print(f"[energy-sweep] idle_power_w={idle_w:.4f}")

    rows: list[dict[str, float | int | str]] = []
    peaks: dict[int, dict[str, float | int]] = {}
    baseline_efficiency: float | None = None

    for heads, batches in sweeps:
        for batch in batches:
            sample_path = sample_dir / f"work_h{heads}_b{batch}.txt"
            try:
                result, workload_w, iterations = measure_one(
                    backend=args.backend,
                    n=args.n,
                    k=args.k,
                    heads=heads,
                    batch=batch,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                    power_seconds=args.power_seconds,
                    loop_ms=args.loop_ms,
                    gpu_index=args.gpu_index,
                    sample_path=sample_path,
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
            dynamic_w = workload_w - idle_w
            energy_mj = dynamic_w * latency_ms
            ops = relative_ops(heads, batch)
            efficiency = ops / energy_mj if energy_mj > 0 else float("nan")
            if baseline_efficiency is None:
                if heads != 1 or batch != 1:
                    raise SystemExit("First measurement must be H=1 B=1 for normalization.")
                baseline_efficiency = efficiency
            efficiency_norm = efficiency / baseline_efficiency
            energy_per_op = energy_mj / ops

            row = {
                "backend": result.backend,
                "n": result.n,
                "k": result.k,
                "heads": heads,
                "batch": batch,
                "latency_ms_mean": latency_ms,
                "idle_power_w": idle_w,
                "workload_power_w": workload_w,
                "dynamic_power_w": dynamic_w,
                "power_loop_iterations": iterations,
                "relative_ops": ops,
                "energy_mj": energy_mj,
                "energy_per_op_mj": energy_per_op,
                "efficiency": efficiency,
                "efficiency_norm": efficiency_norm,
            }
            rows.append(row)
            print(
                f"ENERGY heads={heads} batch={batch} "
                f"dyn_w={dynamic_w:.2f} energy_mj={energy_mj:.4f} "
                f"eff_norm={efficiency_norm:.3f}"
            )

            peak = peaks.get(heads)
            if peak is None or efficiency_norm > peak["efficiency_norm"]:
                peaks[heads] = {
                    "batch": batch,
                    "efficiency_norm": efficiency_norm,
                    "dynamic_power_w": dynamic_w,
                    "energy_mj": energy_mj,
                    "latency_ms_mean": latency_ms,
                }

            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {args.out}")
    print("Peak energy efficiency (normalized to H=1 B=1):")
    for heads in sorted(peaks):
        peak = peaks[heads]
        print(
            f"  H={heads}: B={peak['batch']}  "
            f"eff_norm={peak['efficiency_norm']:.3f}  "
            f"dyn_w={peak['dynamic_power_w']:.2f}  "
            f"energy_mj={peak['energy_mj']:.4f}"
        )


if __name__ == "__main__":
    main()

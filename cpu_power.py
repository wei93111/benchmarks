#!/usr/bin/env python3
"""Measure CPU package power with Linux RAPL energy counters."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_RAPL = Path("/sys/class/powercap/intel-rapl/intel-rapl:0")
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rapl-dir", type=Path, default=DEFAULT_RAPL)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "power" / "cpu_rapl_power_summary.txt",
        help="Summary output path.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after '--'.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing workload command after '--'")
    return args


def read_int(path: Path) -> int:
    return int(path.read_text().strip())


def read_energy_uj(rapl_dir: Path) -> int:
    return read_int(rapl_dir / "energy_uj")


def read_max_energy_uj(rapl_dir: Path) -> int | None:
    path = rapl_dir / "max_energy_range_uj"
    if not path.exists():
        return None
    return read_int(path)


def energy_delta_uj(start: int, end: int, max_range: int | None) -> int:
    if end >= start:
        return end - start
    if max_range is None:
        raise RuntimeError("RAPL energy counter wrapped but max_energy_range_uj is unavailable")
    return (max_range - start) + end


def measure_sleep_power(rapl_dir: Path, seconds: float) -> float:
    max_range = read_max_energy_uj(rapl_dir)
    e0 = read_energy_uj(rapl_dir)
    t0 = time.perf_counter()
    time.sleep(seconds)
    t1 = time.perf_counter()
    e1 = read_energy_uj(rapl_dir)
    return energy_delta_uj(e0, e1, max_range) / 1_000_000.0 / (t1 - t0)


def measure_command_power(rapl_dir: Path, command: list[str]) -> tuple[float, int]:
    max_range = read_max_energy_uj(rapl_dir)
    e0 = read_energy_uj(rapl_dir)
    t0 = time.perf_counter()
    proc = subprocess.run(command)
    t1 = time.perf_counter()
    e1 = read_energy_uj(rapl_dir)
    power_w = energy_delta_uj(e0, e1, max_range) / 1_000_000.0 / (t1 - t0)
    return power_w, proc.returncode


def main() -> None:
    args = parse_args()
    energy_path = args.rapl_dir / "energy_uj"
    if not energy_path.exists():
        raise SystemExit(f"RAPL energy counter not found: {energy_path}")

    try:
        print(f"[cpu-power] measuring idle package power for {args.idle_seconds:.1f}s...")
        idle_w = measure_sleep_power(args.rapl_dir, args.idle_seconds)

        print("[cpu-power] measuring workload package power...")
        workload_w, returncode = measure_command_power(args.rapl_dir, args.command)
    except PermissionError as exc:
        raise SystemExit(
            f"Permission denied reading RAPL counter {energy_path}. "
            "Run this helper with sudo or relax read permissions for the RAPL energy counter."
        ) from exc

    dynamic_w = workload_w - idle_w
    summary = (
        f"idle_power_w={idle_w:.6f}\n"
        f"workload_power_w={workload_w:.6f}\n"
        f"dynamic_power_w={dynamic_w:.6f}\n"
        f"rapl_dir={args.rapl_dir}\n"
        f"returncode={returncode}\n"
    )
    print(summary, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary)
    sys.exit(returncode)


if __name__ == "__main__":
    main()

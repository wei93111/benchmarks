#!/usr/bin/env python3
"""Create clean Triton power and dynamic-energy CSV summaries."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CONFIG_RE = re.compile(r"gpu_n(?P<n>\d+)_k(?P<k>\d+)$")


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def read_latency(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["n"]), int(row["k"])): row for row in rows}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[kernel-triton] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    args = parser.parse_args()

    latency_rows = read_latency(args.root / "gpu_latency.csv")
    power_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []

    summaries = sorted(
        (args.root / "power").glob("gpu_n*_k*/gpu_power_summary.txt"),
        key=lambda path: (
            int(CONFIG_RE.match(path.parent.name).group("n")),
            int(CONFIG_RE.match(path.parent.name).group("k")),
        ),
    )
    for summary_path in summaries:
        match = CONFIG_RE.match(summary_path.parent.name)
        if match is None:
            continue
        n = int(match.group("n"))
        k = int(match.group("k"))
        summary = read_key_values(summary_path)
        latency = latency_rows.get((n, k), {})
        device = latency.get("device", "")
        idle = float(summary["idle_power_w"])
        workload = float(summary["workload_power_w"])
        dynamic = float(summary["dynamic_power_w"])

        power_rows.append(
            {
                "backend": "triton",
                "n": n,
                "k": k,
                "levels": 4,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "device": device,
                "idle_power_w": f"{idle:.4f}",
                "workload_power_w": f"{workload:.4f}",
                "dynamic_power_w": f"{dynamic:.4f}",
            }
        )

        if latency:
            latency_ms = float(latency["latency_ms_mean"])
            energy_rows.append(
                {
                    "backend": "triton",
                    "n": n,
                    "k": k,
                    "heads": args.heads,
                    "head_dim": args.head_dim,
                    "latency_ms": f"{latency_ms:.6f}",
                    "dynamic_power_w": f"{dynamic:.4f}",
                    "dynamic_energy_mj": f"{dynamic * latency_ms:.6f}",
                }
            )

    if power_rows:
        write_csv(
            args.root / "gpu_power.csv",
            [
                "backend",
                "n",
                "k",
                "levels",
                "heads",
                "head_dim",
                "device",
                "idle_power_w",
                "workload_power_w",
                "dynamic_power_w",
            ],
            power_rows,
        )
    if energy_rows:
        write_csv(
            args.root / "gpu_energy.csv",
            [
                "backend",
                "n",
                "k",
                "heads",
                "head_dim",
                "latency_ms",
                "dynamic_power_w",
                "dynamic_energy_mj",
            ],
            energy_rows,
        )


if __name__ == "__main__":
    main()

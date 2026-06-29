#!/usr/bin/env python3
"""Measure CPU package power with Intel PCM CSV energy samples."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"
DEFAULT_SAMPLE_INTERVAL = 0.5
PLATFORM_DESCRIPTION = (
    "AWS c6i.metal; Intel Xeon Platinum 8375C Ice Lake-SP; "
    "2 sockets, 32 physical cores/socket, 64 physical cores total, "
    "128 logical cores with 2-way SMT, 2.90 GHz base"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pcm-bin",
        type=Path,
        default=None,
        help="Path to Intel PCM binary. Defaults to $PCM_BIN, ./bin/pcm, or pcm on PATH.",
    )
    parser.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument(
        "--sudo-pcm",
        action="store_true",
        help="Run only the Intel PCM process through sudo -E. The workload stays in the current user environment.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "power" / "cpu_pcm_power_summary.txt",
        help="Summary output path.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after '--'.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing workload command after '--'")
    return args


def resolve_pcm_bin(requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    if os.environ.get("PCM_BIN"):
        candidates.append(Path(os.environ["PCM_BIN"]))
    candidates.extend(
        [
            Path("./bin/pcm"),
            BENCHMARK_DIR / "bin" / "pcm",
            BENCHMARK_DIR.parent / "bin" / "pcm",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    from_path = shutil.which("pcm")
    if from_path:
        return Path(from_path)

    searched = ", ".join(str(path) for path in candidates)
    raise SystemExit(
        "Intel PCM binary not found. Pass --pcm-bin /path/to/pcm or set PCM_BIN. "
        f"Searched: {searched}, PATH"
    )


def normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def to_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def start_pcm(
    pcm_bin: Path,
    interval: float,
    csv_path: Path,
    log_path: Path,
    sudo_pcm: bool,
) -> subprocess.Popen:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w")
    command = [str(pcm_bin), str(interval), f"-csv={csv_path}"]
    if sudo_pcm:
        command = ["sudo", "-E", *command]
    return subprocess.Popen(
        command,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )


def stop_pcm(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_pcm_window(
    *,
    pcm_bin: Path,
    interval: float,
    csv_path: Path,
    log_path: Path,
    sudo_pcm: bool,
    seconds: float | None = None,
    command: list[str] | None = None,
) -> int:
    proc = start_pcm(pcm_bin, interval, csv_path, log_path, sudo_pcm)
    try:
        # Give PCM a moment to initialize counters before the measured workload starts.
        time.sleep(min(0.2, interval))
        if command is None:
            assert seconds is not None
            time.sleep(seconds)
            returncode = 0
        else:
            returncode = subprocess.run(command).returncode
    finally:
        stop_pcm(proc)
    return returncode


def read_csv_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        raise RuntimeError(f"PCM CSV was not created: {path}")
    text = path.read_text(errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"PCM CSV is empty: {path}")
    sample = "\n".join(lines[:10])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    except csv.Error:
        dialect = csv.excel
    return [row for row in csv.reader(lines, dialect) if any(cell.strip() for cell in row)]


def is_proc_energy_header(cell: str) -> bool:
    name = normalize(cell)
    if "dram" in name or "dimm" in name:
        return False
    if "joule" in name and "energy" in name and "proc" in name:
        return True
    return name in {"cpuenergy", "cpuenergyjoules", "procenergy", "procenergyjoules"}


def is_socket_row(cell: str) -> bool:
    name = normalize(cell)
    return name.startswith("skt") or name.startswith("socket") or name.isdigit()


def is_system_row(cell: str) -> bool:
    name = normalize(cell)
    return name in {"system", "total", "sys"}


def extract_package_energy_samples(csv_path: Path) -> list[float]:
    """Return total package Joules per PCM sampling interval.

    PCM versions differ in CSV layout. This handles the common long format with
    one row per socket plus a System row, and wide formats with one column per
    socket energy value.
    """
    rows = read_csv_rows(csv_path)

    for header_idx, header in enumerate(rows):
        energy_cols = [idx for idx, cell in enumerate(header) if is_proc_energy_header(cell)]
        if not energy_cols:
            continue

        normalized_header = [normalize(cell) for cell in header]
        socket_col = next(
            (idx for idx, name in enumerate(normalized_header) if name in {"skt", "socket"}),
            None,
        )
        date_col = next((idx for idx, name in enumerate(normalized_header) if name == "date"), None)
        time_col = next((idx for idx, name in enumerate(normalized_header) if name == "time"), None)

        if socket_col is not None and len(energy_cols) == 1:
            energy_col = energy_cols[0]
            grouped: dict[tuple[str, ...], float] = {}
            sequence = 0
            for row in rows[header_idx + 1 :]:
                if len(row) <= max(socket_col, energy_col):
                    continue
                socket_label = row[socket_col].strip()
                if is_system_row(socket_label) or not is_socket_row(socket_label):
                    continue
                value = to_float(row[energy_col])
                if value is None:
                    continue
                if date_col is not None and time_col is not None and len(row) > max(date_col, time_col):
                    key = (row[date_col].strip(), row[time_col].strip())
                else:
                    # Fallback for PCM variants without timestamps: start a new sample
                    # when the first socket appears again.
                    if normalize(socket_label) in {"0", "skt0", "socket0"}:
                        sequence += 1
                    key = (str(sequence),)
                grouped[key] = grouped.get(key, 0.0) + value
            samples = list(grouped.values())
            if samples:
                return samples

        socket_energy_cols = [
            idx
            for idx in energy_cols
            if "system" not in normalized_header[idx] and "total" not in normalized_header[idx]
        ]
        if socket_energy_cols:
            samples = []
            for row in rows[header_idx + 1 :]:
                if len(row) <= max(socket_energy_cols):
                    continue
                values = [to_float(row[idx]) for idx in socket_energy_cols]
                if all(value is not None for value in values):
                    samples.append(sum(value for value in values if value is not None))
            if samples:
                return samples

    raise RuntimeError(
        "Could not find PCM processor energy columns. Expected fields like "
        "'CPU energy', 'Proc Energy (Joules)', or 'Proc_Energy_Joules'. "
        f"Inspect raw CSV: {csv_path}"
    )


def summarize_power(csv_path: Path, interval: float) -> tuple[float, float, int]:
    energy_samples = extract_package_energy_samples(csv_path)
    powers = [joules / interval for joules in energy_samples]
    mean_power = sum(powers) / len(powers)
    mean_joules = sum(energy_samples) / len(energy_samples)
    return mean_power, mean_joules, len(energy_samples)


def main() -> None:
    args = parse_args()
    pcm_bin = resolve_pcm_bin(args.pcm_bin)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    idle_csv = args.out.with_name("idle_pcm.csv")
    workload_csv = args.out.with_name("workload_pcm.csv")
    idle_log = args.out.with_name("idle_pcm.log")
    workload_log = args.out.with_name("workload_pcm.log")

    print(f"[cpu-power] platform: {PLATFORM_DESCRIPTION}")
    print(f"[cpu-power] measuring idle package power for {args.idle_seconds:.1f}s with {pcm_bin}...")
    run_pcm_window(
        pcm_bin=pcm_bin,
        interval=args.sample_interval,
        csv_path=idle_csv,
        log_path=idle_log,
        sudo_pcm=args.sudo_pcm,
        seconds=args.idle_seconds,
    )
    idle_w, idle_joules_per_sample, idle_samples = summarize_power(idle_csv, args.sample_interval)

    print("[cpu-power] measuring workload package power with PCM...")
    returncode = run_pcm_window(
        pcm_bin=pcm_bin,
        interval=args.sample_interval,
        csv_path=workload_csv,
        log_path=workload_log,
        sudo_pcm=args.sudo_pcm,
        command=args.command,
    )
    if returncode != 0 and not workload_csv.exists():
        raise SystemExit(
            f"Workload command failed with return code {returncode} before PCM wrote {workload_csv}. "
            "Check the workload error above and rerun after fixing it."
        )
    workload_w, workload_joules_per_sample, workload_samples = summarize_power(
        workload_csv, args.sample_interval
    )

    dynamic_w = workload_w - idle_w
    summary = (
        f"platform={PLATFORM_DESCRIPTION}\n"
        f"pcm_bin={pcm_bin}\n"
        f"sudo_pcm={args.sudo_pcm}\n"
        f"sample_interval_s={args.sample_interval:.6f}\n"
        f"idle_power_w={idle_w:.6f}\n"
        f"workload_power_w={workload_w:.6f}\n"
        f"dynamic_power_w={dynamic_w:.6f}\n"
        f"idle_joules_per_sample={idle_joules_per_sample:.6f}\n"
        f"workload_joules_per_sample={workload_joules_per_sample:.6f}\n"
        f"idle_samples={idle_samples}\n"
        f"workload_samples={workload_samples}\n"
        f"idle_csv={idle_csv}\n"
        f"workload_csv={workload_csv}\n"
        f"idle_log={idle_log}\n"
        f"workload_log={workload_log}\n"
        "energy_scope=total_package_skt0_plus_skt1\n"
        f"returncode={returncode}\n"
    )
    print(summary, end="")
    args.out.write_text(summary)
    sys.exit(returncode)


if __name__ == "__main__":
    main()

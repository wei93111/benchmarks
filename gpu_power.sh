#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gpu_power.sh [--out-dir DIR] [--idle-seconds SEC] [--loop-ms MS] -- COMMAND...

Measures GPU dynamic power using nvidia-smi:
  dynamic_W = steady_workload_power_W - settled_idle_power_W

Example:
  benchmarks/gpu_power.sh \
    --out-dir benchmarks/results/power/gpu_n4096_k8 \
    -- \
    python3 benchmarks/gpu_speed.py --n 4096 --k 8
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR="$SCRIPT_DIR/results/power/gpu"
IDLE_SECONDS=60
LOOP_MS=100
IDLE_START_FRACTION=0.50
IDLE_END_FRACTION=1.00
WORK_START_FRACTION=0.15
WORK_END_FRACTION=0.85
TRIM_FRACTION=0.15

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --idle-seconds)
      IDLE_SECONDS="$2"
      shift 2
      ;;
    --loop-ms)
      LOOP_MS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
IDLE_FILE="$OUT_DIR/gpu_idle_power_w.txt"
WORK_FILE="$OUT_DIR/gpu_workload_power_w.txt"
SUMMARY_FILE="$OUT_DIR/gpu_power_summary.txt"

power_stats() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
start_fraction = float(sys.argv[2])
end_fraction = float(sys.argv[3])
trim_fraction = float(sys.argv[4])
values = []
for line in path.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    values.append(float(line.split()[0]))
if not values:
    raise SystemExit("no power samples collected")
start = int(len(values) * start_fraction)
end = int(len(values) * end_fraction)
start = max(0, min(start, len(values)))
end = max(start + 1, min(end, len(values)))
window = values[start:end]
sorted_window = sorted(window)
trim_count = int(len(sorted_window) * trim_fraction)
if trim_count and len(sorted_window) > 2 * trim_count:
    kept = sorted_window[trim_count:-trim_count]
else:
    kept = sorted_window
mid = len(sorted_window) // 2
if len(sorted_window) % 2:
    median = sorted_window[mid]
else:
    median = (sorted_window[mid - 1] + sorted_window[mid]) / 2
print(f"power_w={sum(kept) / len(kept)}")
print(f"full_window_power_w={sum(window) / len(window)}")
print(f"median_power_w={median}")
print(f"min_power_w={min(window)}")
print(f"max_power_w={max(window)}")
print(f"samples={len(values)}")
print(f"window_samples={len(window)}")
print(f"samples_kept={len(kept)}")
print(f"window_start_index={start}")
print(f"window_end_index={end}")
PY
}

echo "[gpu-power] sampling idle power for ${IDLE_SECONDS}s..."
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms="$LOOP_MS" > "$IDLE_FILE" &
SMI_PID=$!
sleep "$IDLE_SECONDS"
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

IDLE_STATS=$(power_stats "$IDLE_FILE" "$IDLE_START_FRACTION" "$IDLE_END_FRACTION" "$TRIM_FRACTION")
eval "$(printf '%s\n' "$IDLE_STATS" | sed 's/^/IDLE_/')"

echo "[gpu-power] sampling workload power..."
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms="$LOOP_MS" > "$WORK_FILE" &
SMI_PID=$!
set +e
"$@"
WORK_STATUS=$?
set -e
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

WORK_STATS=$(power_stats "$WORK_FILE" "$WORK_START_FRACTION" "$WORK_END_FRACTION" "$TRIM_FRACTION")
eval "$(printf '%s\n' "$WORK_STATS" | sed 's/^/WORK_/')"
DYNAMIC_W=$(python3 - "$IDLE_power_w" "$WORK_power_w" <<'PY'
import sys
idle = float(sys.argv[1])
work = float(sys.argv[2])
print(work - idle)
PY
)

{
  echo "sample_interval_ms=$LOOP_MS"
  echo "trim_fraction=$TRIM_FRACTION"
  echo "idle_power_w=$IDLE_power_w"
  echo "workload_power_w=$WORK_power_w"
  echo "dynamic_power_w=$DYNAMIC_W"
  echo "idle_average_window_fraction=${IDLE_START_FRACTION}:${IDLE_END_FRACTION}"
  echo "workload_average_window_fraction=${WORK_START_FRACTION}:${WORK_END_FRACTION}"
  echo "idle_full_window_power_w=$IDLE_full_window_power_w"
  echo "idle_median_power_w=$IDLE_median_power_w"
  echo "idle_min_power_w=$IDLE_min_power_w"
  echo "idle_max_power_w=$IDLE_max_power_w"
  echo "idle_samples=$IDLE_samples"
  echo "idle_window_samples=$IDLE_window_samples"
  echo "idle_samples_kept=$IDLE_samples_kept"
  echo "idle_window_start_index=$IDLE_window_start_index"
  echo "idle_window_end_index=$IDLE_window_end_index"
  echo "workload_full_window_power_w=$WORK_full_window_power_w"
  echo "workload_median_power_w=$WORK_median_power_w"
  echo "workload_min_power_w=$WORK_min_power_w"
  echo "workload_max_power_w=$WORK_max_power_w"
  echo "workload_samples=$WORK_samples"
  echo "workload_window_samples=$WORK_window_samples"
  echo "workload_samples_kept=$WORK_samples_kept"
  echo "workload_window_start_index=$WORK_window_start_index"
  echo "workload_window_end_index=$WORK_window_end_index"
  echo "idle_samples_file=$IDLE_FILE"
  echo "workload_samples_file=$WORK_FILE"
} | tee "$SUMMARY_FILE"

exit "$WORK_STATUS"

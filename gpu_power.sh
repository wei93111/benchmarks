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
IDLE_SECONDS=10
LOOP_MS=100
IDLE_START_FRACTION=0.50
IDLE_END_FRACTION=1.00
WORK_START_FRACTION=0.15
WORK_END_FRACTION=0.85

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

average_power() {
  python3 - "$1" "$2" "$3" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
start_fraction = float(sys.argv[2])
end_fraction = float(sys.argv[3])
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
print(sum(window) / len(window))
PY
}

echo "[gpu-power] sampling idle power for ${IDLE_SECONDS}s..."
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms="$LOOP_MS" > "$IDLE_FILE" &
SMI_PID=$!
sleep "$IDLE_SECONDS"
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

IDLE_W=$(average_power "$IDLE_FILE" "$IDLE_START_FRACTION" "$IDLE_END_FRACTION")
IDLE_FULL_W=$(average_power "$IDLE_FILE" 0 1)

echo "[gpu-power] sampling workload power..."
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms="$LOOP_MS" > "$WORK_FILE" &
SMI_PID=$!
set +e
"$@"
WORK_STATUS=$?
set -e
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

WORK_W=$(average_power "$WORK_FILE" "$WORK_START_FRACTION" "$WORK_END_FRACTION")
WORK_FULL_W=$(average_power "$WORK_FILE" 0 1)
DYNAMIC_W=$(python3 - "$IDLE_W" "$WORK_W" <<'PY'
import sys
idle = float(sys.argv[1])
work = float(sys.argv[2])
print(work - idle)
PY
)

{
  echo "idle_power_w=$IDLE_W"
  echo "workload_power_w=$WORK_W"
  echo "dynamic_power_w=$DYNAMIC_W"
  echo "idle_full_window_power_w=$IDLE_FULL_W"
  echo "workload_full_window_power_w=$WORK_FULL_W"
  echo "idle_average_window_fraction=${IDLE_START_FRACTION}:${IDLE_END_FRACTION}"
  echo "workload_average_window_fraction=${WORK_START_FRACTION}:${WORK_END_FRACTION}"
  echo "idle_samples=$IDLE_FILE"
  echo "workload_samples=$WORK_FILE"
} | tee "$SUMMARY_FILE"

exit "$WORK_STATUS"

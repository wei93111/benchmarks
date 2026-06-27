#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gpu_power.sh [--out-dir DIR] [--idle-seconds SEC] [--loop-ms MS] -- COMMAND...

Measures GPU dynamic power using nvidia-smi:
  dynamic_W = avg_workload_power_W - avg_idle_power_W

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
  python3 - "$1" <<'PY'
import pathlib
import sys

values = []
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    values.append(float(line.split()[0]))
if not values:
    raise SystemExit("no power samples collected")
print(sum(values) / len(values))
PY
}

echo "[gpu-power] sampling idle power for ${IDLE_SECONDS}s..."
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms="$LOOP_MS" > "$IDLE_FILE" &
SMI_PID=$!
sleep "$IDLE_SECONDS"
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

IDLE_W=$(average_power "$IDLE_FILE")

echo "[gpu-power] sampling workload power..."
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms="$LOOP_MS" > "$WORK_FILE" &
SMI_PID=$!
set +e
"$@"
WORK_STATUS=$?
set -e
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

WORK_W=$(average_power "$WORK_FILE")
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
  echo "idle_samples=$IDLE_FILE"
  echo "workload_samples=$WORK_FILE"
} | tee "$SUMMARY_FILE"

exit "$WORK_STATUS"

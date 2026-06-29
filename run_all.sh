#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

N_VALUES=(1024 4096 16384 65536)
K_VALUES=(4 8 16)

LEVELS=4
HEADS=1
HEAD_DIM=64
WARMUP=100
ITERS=1000
POWER_LOOP_SECONDS=45

usage() {
  cat <<EOF
Usage:
  benchmarks/run_all.sh [options]

Runs the default QuadTree attention benchmark sweep:
  n = ${N_VALUES[*]}
  k = ${K_VALUES[*]}
  levels=$LEVELS heads=$HEADS head_dim=$HEAD_DIM warmup=$WARMUP iters=$ITERS

Options:
  --speed-only       Run GPU+CPU latency only
  --power-only       Run GPU+CPU power only
  --gpu-only         Run GPU latency+power only
  --cpu-only         Run CPU latency+power only
  --gpu-speed-only   Run GPU latency only
  --cpu-speed-only   Run CPU latency only
  --gpu-power-only   Run GPU power only
  --cpu-power-only   Run CPU power only
  --no-sudo          Do not use sudo for CPU PCM power

EOF
}

RUN_SPEED=1
RUN_POWER=1
RUN_GPU=1
RUN_CPU=1
USE_SUDO=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --speed-only)
      RUN_POWER=0
      shift
      ;;
    --power-only)
      RUN_SPEED=0
      shift
      ;;
    --gpu-only)
      RUN_CPU=0
      shift
      ;;
    --cpu-only)
      RUN_GPU=0
      shift
      ;;
    --gpu-speed-only)
      RUN_SPEED=1
      RUN_POWER=0
      RUN_GPU=1
      RUN_CPU=0
      shift
      ;;
    --cpu-speed-only)
      RUN_SPEED=1
      RUN_POWER=0
      RUN_GPU=0
      RUN_CPU=1
      shift
      ;;
    --gpu-power-only)
      RUN_SPEED=0
      RUN_POWER=1
      RUN_GPU=1
      RUN_CPU=0
      shift
      ;;
    --cpu-power-only)
      RUN_SPEED=0
      RUN_POWER=1
      RUN_GPU=0
      RUN_CPU=1
      shift
      ;;
    --no-sudo)
      USE_SUDO=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"
mkdir -p "$SCRIPT_DIR/results"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

if [[ "$RUN_SPEED" -eq 1 && "$RUN_GPU" -eq 1 ]]; then
  echo "[run-all] GPU latency sweep"
  python3 "$SCRIPT_DIR/gpu_speed.py" \
    --n "${N_VALUES[@]}" \
    --k "${K_VALUES[@]}" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --out "$SCRIPT_DIR/results/gpu_latency.csv"
fi

if [[ "$RUN_SPEED" -eq 1 && "$RUN_CPU" -eq 1 ]]; then
  echo "[run-all] CPU latency sweep"
  python3 "$SCRIPT_DIR/cpu_speed.py" \
    --n "${N_VALUES[@]}" \
    --k "${K_VALUES[@]}" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --out "$SCRIPT_DIR/results/cpu_latency.csv"
fi

if [[ "$RUN_POWER" -eq 1 ]]; then
  for n in "${N_VALUES[@]}"; do
    for k in "${K_VALUES[@]}"; do
      if [[ "$RUN_GPU" -eq 1 ]]; then
        echo "[run-all] GPU power n=$n k=$k"
        "$SCRIPT_DIR/gpu_power.sh" \
          --out-dir "$SCRIPT_DIR/results/power/gpu_n${n}_k${k}" \
          -- \
          python3 -c "from qt_bench import run_single_power_loop, POWER_LOOP_SECONDS; run_single_power_loop(backend='cuda_ref', n=$n, k=$k, seconds=POWER_LOOP_SECONDS)"
      fi

      if [[ "$RUN_CPU" -eq 1 ]]; then
        echo "[run-all] CPU power n=$n k=$k"
        CPU_POWER_CMD=(
          python3 "$SCRIPT_DIR/cpu_power.py"
          --out "$SCRIPT_DIR/results/power/cpu_n${n}_k${k}/cpu_pcm_power_summary.txt"
          --
          python3 -c "from qt_bench import run_single_power_loop, POWER_LOOP_SECONDS; run_single_power_loop(backend='torch_cpu', n=$n, k=$k, seconds=POWER_LOOP_SECONDS)"
        )
        if [[ "$USE_SUDO" -eq 1 ]]; then
          sudo -E "${CPU_POWER_CMD[@]}"
        else
          "${CPU_POWER_CMD[@]}"
        fi
      fi
    done
  done
fi

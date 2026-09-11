#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

N_VALUES=(1024 4096 16384 65536)
K_VALUES=(4 8 16)
HEADS=1
HEAD_DIM=64
BATCH=1
PRECISION=fp32
WARMUP=100
ITERS=1000
POWER_LOOP_SECONDS=45
OUT_ROOT=""
LATENCY_CSV=""
RUN_LATENCY=1
RUN_POWER=1

usage() {
  cat <<EOF
Usage:
  benchmarks/run_all_triton.sh [options]

Runs the fused Triton GPU baseline for:
  n = ${N_VALUES[*]}
  k = ${K_VALUES[*]}
  heads=$HEADS batch=$BATCH precision=$PRECISION head_dim=$HEAD_DIM warmup=$WARMUP iters=$ITERS

Options:
  --heads N          Number of attention heads: 1 or 8 (default: $HEADS)
  --batch N          Input batch size (default: $BATCH)
  --precision MODE   fp32 (default) or int8-fp16
  --head-dim N       Per-head dimension; currently must be 64 (default: $HEAD_DIM)
  --out-root DIR     Output root (default: results/triton_results_heads<HEADS>[_batch<N>])
  --latency-csv FILE Latency CSV for energy (default: <out-root>/gpu_latency.csv)
  --latency-only     Run latency sweep only
  --power-only       Run power (+energy summary) only; does not remeasure latency
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --heads)
      HEADS="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --head-dim)
      HEAD_DIM="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --latency-csv)
      LATENCY_CSV="$2"
      shift 2
      ;;
    --latency-only|--speed-only)
      RUN_POWER=0
      shift
      ;;
    --power-only)
      RUN_LATENCY=0
      shift
      ;;
    -h|--help)
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

if [[ "$HEADS" != "1" && "$HEADS" != "8" ]]; then
  echo "Triton baseline currently supports --heads 1 or 8." >&2
  exit 2
fi
if [[ "$HEAD_DIM" != "64" ]]; then
  echo "Triton baseline currently requires --head-dim 64." >&2
  exit 2
fi
if [[ "$BATCH" -lt 1 ]]; then
  echo "Batch size must be a positive integer, got $BATCH." >&2
  exit 2
fi
if [[ "$PRECISION" != "fp32" && "$PRECISION" != "int8-fp16" ]]; then
  echo "Precision must be fp32 or int8-fp16, got $PRECISION." >&2
  exit 2
fi
if [[ -z "$OUT_ROOT" ]]; then
  PRECISION_SUFFIX=""
  if [[ "$PRECISION" != "fp32" ]]; then
    PRECISION_SUFFIX="_${PRECISION}"
  fi
  if [[ "$BATCH" -eq 1 ]]; then
    OUT_ROOT="$SCRIPT_DIR/results/triton_results_heads${HEADS}${PRECISION_SUFFIX}"
  else
    OUT_ROOT="$SCRIPT_DIR/results/triton_results_heads${HEADS}_batch${BATCH}${PRECISION_SUFFIX}"
  fi
fi

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

python3 - <<'PY'
import torch
import triton
from kernel_triton import coarse_attention, fine_attention

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable.")
print(
    f"[run-all-triton] torch={torch.__version__} triton={triton.__version__} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
PY

if [[ "$RUN_LATENCY" -eq 1 ]]; then
  echo "[run-all-triton] latency sweep"
  rm -f "$OUT_ROOT/gpu_latency.csv"
  for n in "${N_VALUES[@]}"; do
    for k in "${K_VALUES[@]}"; do
      echo "[run-all-triton] latency n=$n k=$k"
      python3 "$SCRIPT_DIR/gpu_speed.py" \
        --backend triton \
        --n "$n" \
        --k "$k" \
        --heads "$HEADS" \
        --head-dim "$HEAD_DIM" \
        --batch "$BATCH" \
        --precision "$PRECISION" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --out "$OUT_ROOT/gpu_latency.csv"
    done
  done
fi

if [[ "$RUN_POWER" -eq 1 ]]; then
  for n in "${N_VALUES[@]}"; do
    for k in "${K_VALUES[@]}"; do
      echo "[run-all-triton] power n=$n k=$k"
      "$SCRIPT_DIR/gpu_power.sh" \
        --out-dir "$OUT_ROOT/power/gpu_n${n}_k${k}" \
        -- \
        python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from qt_bench import run_single_power_loop, POWER_LOOP_SECONDS; run_single_power_loop(backend='triton', n=$n, k=$k, heads=$HEADS, head_dim=$HEAD_DIM, batch=$BATCH, precision='$PRECISION', seconds=POWER_LOOP_SECONDS)"
    done
  done
fi

python3 "$SCRIPT_DIR/kernel_triton/summarize_results.py" \
  --root "$OUT_ROOT" \
  --heads "$HEADS" \
  --head-dim "$HEAD_DIM" \
  --batch "$BATCH" \
  --precision "$PRECISION" \
  ${LATENCY_CSV:+--latency-csv "$LATENCY_CSV"}

# QuadTree Attention Benchmarks

This folder measures GPU/CPU speed and power for the QuadTree Attention-B core.
The benchmark is attention-only: it excludes full model stacks, projections,
MLPs, data loading, and host-device transfers.

## What Is Measured

Defaults:

```text
n = 1024, 4096, 16384, 65536
k = 4, 8, 16
levels = 4
heads = 1
head_dim = 64
channels = 64
LePE = enabled
warmup = 100
iters = 1000
trim = discard lowest/highest 15%
```

The GPU path uses the custom CUDA kernels:

```text
score_computation_cuda
value_aggregation_cuda
```

The CPU path uses PyTorch fallback implementations of the same score and value
aggregation operations.

## Setup

Activate the environment you want to measure:

```bash
conda activate cuda121
```

Install regular Python dependencies if needed:

```bash
python3 - <<'PY'
import torch
import einops
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("einops OK")
PY
```

## Build CUDA Kernels

GPU benchmarks require the custom extension modules. Build them from the copied
sources in `benchmarks/kernel/`:

```bash
cd benchmarks/kernel
python3 setup.py install
cd ../..
```

Or:

```bash
benchmarks/kernel/build.sh
```

Verify the modules import:

```bash
python3 - <<'PY'
import score_computation_cuda
import value_aggregation_cuda
print("CUDA extensions OK")
PY
```

## Run All Evaluations

This runs GPU speed, CPU speed, GPU power, and CPU power for the full default
`(n, k)` sweep:

```bash
benchmarks/run_all.sh
```

CPU power uses RAPL and may require `sudo`. If RAPL permissions are already
configured, use:

```bash
benchmarks/run_all.sh --no-sudo
```

Useful selectors:

```bash
benchmarks/run_all.sh --gpu-speed-only
benchmarks/run_all.sh --cpu-speed-only
benchmarks/run_all.sh --gpu-power-only
benchmarks/run_all.sh --cpu-power-only
benchmarks/run_all.sh --gpu-only
benchmarks/run_all.sh --cpu-only
```

Use `--gpu-power-only` when CPU RAPL permissions are unavailable and you only
want GPU dynamic power.

## 1. GPU Speed

Run the full GPU latency sweep:

```bash
python3 benchmarks/gpu_speed.py
```

Output:

```text
benchmarks/results/gpu_latency.csv
```

Timing method:

- `torch.cuda.Event`
- 100 warmup iterations
- 1000 timed iterations
- discard lowest/highest 15%
- report trimmed mean latency in ms

## 2. CPU Speed

Run the full CPU latency sweep:

```bash
python3 benchmarks/cpu_speed.py
```

Output:

```text
benchmarks/results/cpu_latency.csv
```

Timing method:

- `time.perf_counter()`
- default PyTorch/MKL CPU threading
- 100 warmup iterations
- 1000 timed iterations
- discard lowest/highest 15%
- report trimmed mean latency in ms

## 3. GPU Power / Energy Inputs

Run GPU power for all default configs:

```bash
benchmarks/run_all.sh --gpu-power-only
```

Or for one config:

```bash
benchmarks/gpu_power.sh \
  --out-dir benchmarks/results/power/gpu_n4096_k8 \
  -- \
  python3 -c "from qt_bench import run_single_power_loop, POWER_LOOP_SECONDS; run_single_power_loop(backend='cuda_ref', n=4096, k=8, seconds=POWER_LOOP_SECONDS)"
```

Output per config:

```text
benchmarks/results/power/gpu_n<N>_k<K>/gpu_power_summary.txt
```

Power method:

- sample idle GPU power for 10 seconds with `nvidia-smi`
- run sustained attention workload for 45 seconds
- sample workload GPU power with `nvidia-smi --loop-ms=100`
- dynamic power = workload power - idle power

## 4. CPU Power / Energy Inputs

Run CPU power for all default configs:

```bash
benchmarks/run_all.sh --cpu-power-only
```

If RAPL requires root, keep the default `sudo` behavior. If not:

```bash
benchmarks/run_all.sh --cpu-power-only --no-sudo
```

Output per config:

```text
benchmarks/results/power/cpu_n<N>_k<K>/rapl_power_summary.txt
```

Power method:

- sample idle CPU package energy for 10 seconds with RAPL
- run sustained CPU attention workload for 45 seconds with default threading
- compute workload package power from RAPL energy delta
- dynamic power = workload power - idle power

## Energy Calculation

The scripts collect latency and dynamic power separately. Energy per attention
invocation is:

```text
energy_mJ = dynamic_power_W * latency_ms
```

because:

```text
1 W * 1 ms = 1 mJ
```

## Result Files

Latency:

```text
benchmarks/results/gpu_latency.csv
benchmarks/results/cpu_latency.csv
```

Power:

```text
benchmarks/results/power/gpu_n<N>_k<K>/gpu_power_summary.txt
benchmarks/results/power/cpu_n<N>_k<K>/rapl_power_summary.txt
```

## Notes for Rental Machines

When moving only `benchmarks/` to a new GPU machine:

1. Install PyTorch with CUDA support.
2. Install `einops`.
3. Build kernels with `benchmarks/kernel/build.sh`.
4. Confirm `score_computation_cuda` and `value_aggregation_cuda` import.
5. Run `benchmarks/run_all.sh`.

Report the actual measured platform. For example, results from an RTX 4070 Ti
should not be described as A100 results. To obtain A100 numbers, rerun these
same scripts on A100.

# QuadTree Triton Kernels

This folder contains an inference-only fused Triton baseline for the benchmark's
QuadTree Attention-B GPU path.

The implementation has two kernels:

- `coarse_attention`: dense FlashAttention-style online softmax, weighted-V
  accumulation, and routing top-K in one kernel.
- `fine_attention`: candidate-index generation, sparse QK, softmax, weighted-V
  accumulation, and next-level routing top-K in one kernel.

Each fine-level program processes all four query children of one parent so the
gathered candidate K/V tile is reused across those queries.

The kernels never materialize score, scaled-logit, probability, or repeated
index tensors in global memory. LePE and cross-level message accumulation remain
in `qt_bench.py` and are shared with the original CUDA baseline.

## Supported Configuration

- CUDA GPU
- FP32 inference
- batch size 1
- head dimension 64
- heads 1 or 8
- top-K 4, 8, or 16

## H100 Setup

Create a clean environment on the H100 machine:

```bash
cd benchmarks
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install einops
```

The Linux PyTorch wheel installs its matching Triton dependency. Confirm that
the GPU and dependencies are ready:

```bash
./kernel_triton/build.sh
```

There is no NVCC or extension build. Triton compiles and caches each kernel
during benchmark warmup.

## Small Benchmark

```bash
python3 gpu_speed.py \
  --backend triton --n 1024 --k 4 --heads 1 --warmup 10 --iters 100
```

GPU latency timing batches 10 invocations per CUDA event pair for
`n < 65536`; `n=65536` remains unbatched. The reported value is always
milliseconds per invocation.

## Full Triton Sweep

```bash
./run_all_triton.sh \
  --heads 1 \
  --out-root "$PWD/results/H100_triton_results_heads1"

./run_all_triton.sh \
  --heads 8 \
  --out-root "$PWD/results/H100_triton_results_heads8"
```

Each run produces:

- `gpu_latency.csv`
- per-configuration idle/workload power traces and summaries
- `gpu_power.csv`
- `gpu_energy.csv`, where dynamic energy is
  `dynamic_power_w * latency_ms`

Use `--latency-only` while developing kernels to avoid the long power sweep.
The full power sweep samples 60 seconds of idle power and 45 seconds of workload
power for each of the 12 `(n, K)` configurations, so allow about 21 minutes per
head-count sweep.

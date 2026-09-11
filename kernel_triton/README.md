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
- FP32 inference by default
- optional INT8 QK/SV and INT8 LePE depthwise convolution with INT32
  accumulation, FP16 routing/output, and FP32 online-softmax accumulation
- batch size >= 1
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
  --backend triton --n 1024 --k 4 --heads 1 --batch 1 --warmup 10 --iters 100
```

The default remains the original full-FP32 path. The opt-in quantized-datapath
mode allocates synthetic Q/K/V and LePE weights directly as INT8, so it requires
no separate quantization pass. It uses fixed dummy scales (`Q/K/V scale = 16`,
LePE weight scale = 16, probability scale = 127) and zero INT32 LePE bias.
QK and SV use INT8 Tensor-Core dot products with INT32 results. LePE uses an
INT8 depthwise 3x3 convolution with INT32 accumulation. Routing and messages use
FP16, while numerically sensitive coarse online-softmax state accumulates in
FP32. This mode is intended for architecture-level latency and power
measurements, not accuracy evaluation or deployment.

```bash
python3 gpu_speed.py \
  --backend triton --precision int8-fp16 \
  --n 1024 --k 4 --heads 8 --batch 32 --warmup 10 --iters 100
```

GPU latency timing groups 10 forward invocations per CUDA event pair for
`n < 65536` (including input `--batch` > 1). `n=65536` remains one invocation
per event. The reported value is always milliseconds per forward, and
`--batch` is the NCHW data batch in that forward.

## Full Triton Sweep

```bash
./run_all_triton.sh \
  --heads 1 \
  --batch 1 \
  --out-root "$PWD/results/H100_triton_results_heads1"

./run_all_triton.sh \
  --heads 8 \
  --batch 1 \
  --out-root "$PWD/results/H100_triton_results_heads8"

./run_all_triton.sh \
  --heads 1 \
  --batch 256 \
  --latency-only \
  --out-root "$PWD/results/H100_triton_results_heads1_batch256"
```

`gpu_speed.py` and `run_all_triton.sh` sweep the usual `(n, K)` grid at a fixed `--heads` and `--batch`. Latency uses 100 warmup steps, 1000 timed iterations, and 15% symmetric outlier trim by default.

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

Power and energy only, using an existing latency CSV:

```bash
./run_all_triton.sh \
  --heads 8 \
  --batch 32 \
  --power-only \
  --out-root "$PWD/results/H100_triton_results_h8_b32" \
  --latency-csv "$PWD/results/H100_triton_results_h8_b32/gpu_latency.csv"
```

## INT8 QK/SV + FP16 Sweep

Latency only:

```bash
./run_all_triton.sh \
  --precision int8-fp16 --heads 8 --batch 32 --latency-only \
  --out-root "$PWD/results/H100_triton_int8_fp16_h8_b32"
```

Power only (and energy when the latency CSV is in the same output root):

```bash
./run_all_triton.sh \
  --precision int8-fp16 --heads 8 --batch 32 --power-only \
  --out-root "$PWD/results/H100_triton_int8_fp16_h8_b32"
```

Latency and power:

```bash
./run_all_triton.sh \
  --precision int8-fp16 --heads 8 --batch 32 \
  --out-root "$PWD/results/H100_triton_int8_fp16_h8_b32"
```

# QuadTree CUDA Kernels

This folder contains the custom CUDA extension sources needed by the benchmark
GPU path.

The benchmark imports two Python extension modules:

```text
score_computation_cuda
value_aggregation_cuda
```

Build and install them inside the active Python environment before running GPU
speed or GPU power measurements.

## Build

From the repository root:

```bash
cd benchmarks/kernel
python3 setup.py install
```

Or:

```bash
benchmarks/kernel/build.sh
```

## Requirements

- PyTorch with CUDA support
- CUDA toolkit / `nvcc`
- A compiler compatible with the PyTorch extension build
- `setuptools`

Check the environment with:

```bash
python3 - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("CUDA_HOME:", CUDA_HOME)
PY
nvcc --version
```

## Smoke Test

After building:

```bash
python3 - <<'PY'
import score_computation_cuda
import value_aggregation_cuda
print("score_computation_cuda OK")
print("value_aggregation_cuda OK")
PY
```

Then run a tiny GPU benchmark:

```bash
python3 ../gpu_speed.py --n 1024 --k 4 --warmup 1 --iters 1
```

## Notes

The source is copied from the original QuadTree Attention reference CUDA
implementation. The value aggregation kernel uses CUDA `atomicAdd` directly for
modern PyTorch/CUDA compatibility.

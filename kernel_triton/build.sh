#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
try:
    import torch
    import triton
    import einops
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install the CUDA PyTorch wheel and einops."
    ) from exc

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the driver and CUDA PyTorch installation.")
print(
    f"[kernel-triton] ready: torch={torch.__version__} "
    f"triton={triton.__version__} gpu={torch.cuda.get_device_name(0)}"
)
PY

echo "[kernel-triton] no build is needed; kernels compile during benchmark warmup"

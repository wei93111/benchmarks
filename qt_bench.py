"""Shared QuadTree attention benchmark utilities."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
from einops.einops import rearrange


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "results"

N_VALUES = (1024, 4096, 16384, 65536)
K_VALUES = (4, 8, 16)
LEVELS = 4
HEADS = 1
HEAD_DIM = 64
BATCH = 1
WARMUP = 100
ITERS = 1000
TRIM = 0.15
CUDA_TIMING_BATCH_SIZE = 10
CUDA_UNBATCHED_N = 65536
CPU_THREADS = None
POWER_LOOP_SECONDS = 45.0
LEPE = True
PRECISION_FP32 = "fp32"
PRECISION_INT8_FP16 = "int8-fp16"
PRECISION_VALUES = (PRECISION_FP32, PRECISION_INT8_FP16)


@dataclass(frozen=True)
class BenchConfig:
    backend: str
    n: int
    k: int
    levels: int
    heads: int
    head_dim: int
    batch: int
    precision: str
    channels: int
    side: int
    dtype: str
    device: str
    lepe: bool


@dataclass(frozen=True)
class LatencyResult:
    backend: str
    n: int
    k: int
    levels: int
    heads: int
    head_dim: int
    batch: int
    precision: str
    channels: int
    side: int
    dtype: str
    device: str
    lepe: bool
    warmup: int
    iters: int
    timing_batch_size: int
    timed_batches: int
    trim_fraction: float
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_min: float
    latency_ms_max: float
    samples_kept: int


def score_computation_torch(query: torch.Tensor, key: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """CPU fallback matching the reference CUDA score op."""
    index = index.long()
    bsz, n_query, _, n_heads, _ = query.shape
    topk = index.shape[2]
    b_idx = torch.arange(bsz, device=query.device).view(bsz, 1, 1, 1).expand(bsz, n_query, topk, n_heads)
    h_idx = torch.arange(n_heads, device=query.device).view(1, 1, 1, n_heads).expand(bsz, n_query, topk, n_heads)
    gathered = key[b_idx, index, h_idx]
    return torch.einsum("bnfhd,bnkhd->bnfkh", query, gathered)


def value_aggregation_torch(score: torch.Tensor, value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """CPU fallback matching the reference CUDA value aggregation op."""
    batch, n_query, quad_f, topk, n_heads = score.shape
    score_flat = score.reshape(batch, n_query * quad_f, topk, n_heads)
    index_flat = index.reshape(batch, n_query * quad_f, topk, n_heads).long()
    _, n_flat, _, _ = score_flat.shape

    b_idx = torch.arange(batch, device=score.device).view(batch, 1, 1, 1).expand(batch, n_flat, topk, n_heads)
    h_idx = torch.arange(n_heads, device=score.device).view(1, 1, 1, n_heads).expand(batch, n_flat, topk, n_heads)
    gathered = value[b_idx, index_flat, h_idx]
    out = (score_flat.unsqueeze(-1) * gathered).sum(dim=2)
    return out.view(batch, n_query, quad_f, n_heads, out.shape[-1])


def score_computation_cuda_op(query: torch.Tensor, key: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    try:
        import score_computation_cuda
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing score_computation_cuda. Build/install the CUDA extensions first:\n"
            "  benchmarks/kernel/build.sh"
        ) from exc
    return score_computation_cuda.score_forward(query, key, index)[0]


def value_aggregation_cuda_op(score: torch.Tensor, value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    try:
        import value_aggregation_cuda
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing value_aggregation_cuda. Build/install the CUDA extensions first:\n"
            "  benchmarks/kernel/build.sh"
        ) from exc

    quad_f = score.shape[2]
    score_flat = rearrange(score, "b n f K h -> b (n f) K h")
    index_flat = rearrange(index, "b n f K h -> b (n f) K h")
    batch, n_flat, _, n_heads = score_flat.shape
    head_dim = value.shape[-1]
    output = score.new_zeros([batch, n_flat, n_heads, head_dim]).contiguous()
    value_aggregation_cuda.value_aggregation_forward(score_flat, value, index_flat, output)
    return rearrange(output, "b (n f) h d -> b n f h d", f=quad_f)


def coarse_attention_triton_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from kernel_triton import coarse_attention
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Triton kernel dependencies. Check the environment with:\n"
            "  benchmarks/kernel_triton/build.sh"
        ) from exc
    return coarse_attention(query, key, value, topk)


def coarse_attention_triton_int8_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from kernel_triton import coarse_attention_int8
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Triton kernel dependencies. Check the environment with:\n"
            "  benchmarks/kernel_triton/build.sh"
        ) from exc
    return coarse_attention_int8(query, key, value, topk)


def fine_attention_triton_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    previous_topk_idx: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from kernel_triton import fine_attention
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Triton kernel dependencies. Check the environment with:\n"
            "  benchmarks/kernel_triton/build.sh"
        ) from exc
    return fine_attention(query, key, value, previous_topk_idx, topk)


def fine_attention_triton_int8_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    previous_topk_idx: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from kernel_triton import fine_attention_int8
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Triton kernel dependencies. Check the environment with:\n"
            "  benchmarks/kernel_triton/build.sh"
        ) from exc
    return fine_attention_int8(query, key, value, previous_topk_idx, topk)


def lepe_triton_int8_op(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    try:
        from kernel_triton import depthwise_conv3x3_int8
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Triton kernel dependencies. Check the environment with:\n"
            "  benchmarks/kernel_triton/build.sh"
        ) from exc
    return depthwise_conv3x3_int8(value, weight, bias)


score_computation_op = score_computation_torch
value_aggregation_op = value_aggregation_torch


class QTAttB(nn.Module):
    """Local copy of the reference QuadTree Attention-B attention core."""

    def __init__(
        self,
        nhead,
        dim,
        scale,
        topks=(32, 32, 32, 32),
        use_dropout=False,
        attention_dropout=0.1,
        lepe=False,
        backend="torch_cpu",
        precision=PRECISION_FP32,
    ):
        super().__init__()
        del attention_dropout
        self.use_dropout = use_dropout
        self.topks = list(topks)
        self.nhead = nhead
        self.dim = dim
        self.lepe = lepe
        self.backend = backend
        self.precision = precision
        if lepe and precision == PRECISION_INT8_FP16:
            channels = dim * nhead
            for level in range(scale):
                self.register_buffer(
                    f"lepe_weight_int8_{level}",
                    torch.randint(-16, 17, (channels, 3, 3), dtype=torch.int8),
                )
                self.register_buffer(
                    f"lepe_bias_int32_{level}",
                    torch.zeros(channels, dtype=torch.int32),
                )
        elif lepe:
            self.get_vs = nn.ModuleList(
                [
                    nn.Conv2d(dim * nhead, dim * nhead, kernel_size=3, stride=1, padding=1, groups=dim * nhead)
                    for _ in range(scale)
                ]
            )
        self.register_parameter("weight", nn.Parameter(torch.randn(scale)))

    def process_coarse_level(self, query, key, value, topk):
        if self.backend == "triton":
            if self.precision == PRECISION_INT8_FP16:
                message, topk_idx = coarse_attention_triton_int8_op(
                    query, key, value, topk
                )
            else:
                message, topk_idx = coarse_attention_triton_op(
                    query, key, value, topk
                )
            return None, message, None, topk_idx

        batch, channels, height, width = key.shape
        del height, width
        cur_dim = channels // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(batch, -1, self.nhead, cur_dim)
        value = rearrange(value, "b c h w -> b (h w) c").view(batch, -1, self.nhead, cur_dim)
        query = rearrange(query, "b c h w -> b (h w) c").view(batch, -1, self.nhead, cur_dim)

        qk = torch.einsum("bnhd,bshd->bnsh", query, key)
        logits = qk / cur_dim**0.5
        probs = torch.softmax(logits, dim=-2)
        topk_score, topk_idx = torch.topk(probs, dim=-2, k=topk, largest=True)
        message = torch.einsum("bnsh,bshd->bnhd", probs, value)
        return probs, message, topk_score, topk_idx

    def process_fine_level(self, query, key, value, topk_score, topk_pos, topk_prev, topk, final=False):
        del final
        batch, channels, height, width = key.shape
        cur_dim = channels // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(batch, -1, self.nhead, cur_dim)
        value = rearrange(value, "b c h w -> b (h w) c").view(batch, -1, self.nhead, cur_dim)

        query = query.view(batch, channels, height // 2, 2, width // 2, 2)
        query = rearrange(query, "b c h t1 w t2 -> b (h w) (t1 t2) c").view(
            batch, -1, 4, self.nhead, cur_dim
        )

        topk_pos = topk_pos * 2
        idx_gather = []
        for x in (0, 1):
            for y in (0, 1):
                idx_gather.append((topk_pos[0] + x) * width + topk_pos[1] + y)
        idx = torch.stack(idx_gather, dim=3)

        qk = score_computation_op(query, key.contiguous(), idx.view(batch, -1, topk_prev * 4, self.nhead))
        probs = torch.softmax(qk / cur_dim**0.5, dim=-2)
        probs = probs.reshape(batch, -1, 4, topk_prev * 4, self.nhead)
        idx = idx.view(batch, -1, 1, topk_prev * 4, self.nhead).repeat(1, 1, 4, 1, 1)

        topk_score, topk_idx = torch.topk(probs, dim=-2, k=topk, largest=True)
        message = value_aggregation_op(probs, value.contiguous(), idx)
        topk_idx = torch.gather(idx, index=topk_idx, dim=-2)
        topk_idx = rearrange(topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=height // 2, t1=2)
        topk_score = rearrange(
            topk_score,
            "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh",
            h=height // 2,
            t1=2,
        )
        return probs, message, topk_score, topk_idx

    def process_fine_level_triton(self, query, key, value, previous_topk_idx, topk):
        if self.precision == PRECISION_INT8_FP16:
            message, topk_idx = fine_attention_triton_int8_op(
                query, key, value, previous_topk_idx, topk
            )
        else:
            message, topk_idx = fine_attention_triton_op(
                query, key, value, previous_topk_idx, topk
            )
        return None, message, None, topk_idx

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        del q_mask, kv_mask
        messages = []
        topk = self.topks[0]
        topk_pos = None
        previous_topk_idx = None

        for i, (query, key, value) in enumerate(
            zip(reversed(queries), reversed(keys), reversed(values))
        ):
            if i == 0:
                _, message, topk_score, topk_idx = self.process_coarse_level(query, key, value, topk)
            else:
                topk_prev = topk
                topk = self.topks[i]
                final = i == len(queries) - 1
                if self.backend == "triton":
                    _, message, topk_score, topk_idx = self.process_fine_level_triton(
                        query,
                        key,
                        value,
                        previous_topk_idx,
                        topk,
                    )
                else:
                    _, message, topk_score, topk_idx = self.process_fine_level(
                        query, key, value, topk_score, topk_pos, topk_prev, topk, final
                    )
            messages.append(message)
            if self.backend == "triton":
                previous_topk_idx = topk_idx
            else:
                _, _, height, width = key.shape
                topk_pos = torch.stack([topk_idx // width, topk_idx % width])

        weight = torch.softmax(self.weight, dim=0)
        final_message = 0
        for i, message in enumerate(messages):
            if self.lepe:
                if self.precision == PRECISION_INT8_FP16:
                    lepe = lepe_triton_int8_op(
                        values[-(i + 1)],
                        getattr(self, f"lepe_weight_int8_{i}"),
                        getattr(self, f"lepe_bias_int32_{i}"),
                    )
                else:
                    lepe = self.get_vs[i](values[-(i + 1)])
            if i == 0:
                if self.lepe:
                    lepe = rearrange(lepe, "b (hd d) H W -> b (H W) hd d", hd=self.nhead)
                    final_message = (message + lepe) * weight[i]
                else:
                    final_message = message * weight[i]
            else:
                if self.lepe:
                    lepe = rearrange(
                        lepe,
                        "b (hd d) (H t1) (W t2) -> b (H W) (t1 t2) hd d",
                        hd=self.nhead,
                        t1=2,
                        t2=2,
                    )
                    final_message = final_message.unsqueeze(2) + (message + lepe) * weight[i]
                else:
                    final_message = final_message.unsqueeze(2) + message * weight[i]
                final_message = rearrange(
                    final_message,
                    "b (H W) (t1 t2) h d -> b (H t1 W t2) h d",
                    t1=2,
                    t2=2,
                    H=queries[-i].shape[2],
                )
        return final_message


def configure_backend_ops(backend: str) -> None:
    global score_computation_op, value_aggregation_op
    if backend == "cuda_ref":
        score_computation_op = score_computation_cuda_op
        value_aggregation_op = value_aggregation_cuda_op
    else:
        score_computation_op = score_computation_torch
        value_aggregation_op = value_aggregation_torch


def is_cuda_backend(backend: str) -> bool:
    return backend in {"cuda_ref", "triton"}


def get_device(backend: str) -> torch.device:
    if is_cuda_backend(backend):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA backend requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(name: str, device: torch.device, backend: str) -> torch.dtype:
    if name == "float16":
        if backend in {"cuda_ref", "triton"}:
            raise SystemExit(f"{backend} currently supports float32 benchmarking only.")
        if device.type != "cuda":
            raise SystemExit("float16 benchmarking is only supported on CUDA.")
        return torch.float16
    return torch.float32


def validate_shape(n: int, k: int, levels: int) -> int:
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"n={n} is not a square token count.")
    divisor = 1 << (levels - 1)
    if side % divisor != 0:
        raise ValueError(f"sqrt(n)={side} must be divisible by 2^(levels-1)={divisor}.")
    coarsest_tokens = (side // divisor) ** 2
    if k > coarsest_tokens:
        raise ValueError(f"k={k} exceeds coarsest-level token count {coarsest_tokens} for n={n}.")
    return side


def make_pyramid(
    *,
    side: int,
    levels: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    batch: int = BATCH,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}.")
    queries: list[torch.Tensor] = []
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for level in range(levels):
        level_side = side // (1 << level)
        shape = (batch, channels, level_side, level_side)
        if dtype == torch.int8:
            # Synthetic benchmark tensors are allocated directly in the
            # quantized domain; no setup quantization pass is required.
            def make_int8() -> torch.Tensor:
                return torch.randint(
                    -16,
                    17,
                    shape,
                    device=device,
                    dtype=torch.int8,
                )

            queries.append(make_int8())
            keys.append(make_int8())
            values.append(make_int8())
        else:
            queries.append(torch.randn(shape, device=device, dtype=dtype))
            keys.append(torch.randn(shape, device=device, dtype=dtype))
            values.append(torch.randn(shape, device=device, dtype=dtype))
    return queries, keys, values


def build_case(
    *,
    backend: str,
    n: int,
    k: int,
    levels: int = LEVELS,
    heads: int = HEADS,
    head_dim: int = HEAD_DIM,
    batch: int = BATCH,
    precision: str = PRECISION_FP32,
    dtype_name: str = "float32",
    lepe: bool = LEPE,
    seed: int = 0,
):
    if backend not in {"cuda_ref", "triton", "torch_cpu"}:
        raise ValueError(f"Unknown backend {backend!r}.")
    if precision not in PRECISION_VALUES:
        raise ValueError(f"precision must be one of {PRECISION_VALUES}, got {precision!r}.")
    if precision == PRECISION_INT8_FP16 and backend != "triton":
        raise ValueError("INT8/FP16 precision is supported only by the Triton backend.")
    device = get_device(backend)
    dtype = (
        torch.int8
        if precision == PRECISION_INT8_FP16
        else get_dtype(dtype_name, device, backend)
    )
    side = validate_shape(n, k, levels)
    channels = heads * head_dim
    configure_backend_ops(backend)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    attn = QTAttB(
        nhead=heads,
        dim=head_dim,
        scale=levels,
        topks=[k] * levels,
        lepe=lepe,
        backend=backend,
        precision=precision,
    )
    module_dtype = torch.float16 if precision == PRECISION_INT8_FP16 else dtype
    attn = attn.to(device=device, dtype=module_dtype)
    attn.eval()

    queries, keys, values = make_pyramid(
        side=side,
        levels=levels,
        channels=channels,
        device=device,
        dtype=dtype,
        batch=batch,
    )

    config = BenchConfig(
        backend=backend,
        n=n,
        k=k,
        levels=levels,
        heads=heads,
        head_dim=head_dim,
        batch=batch,
        precision=precision,
        channels=channels,
        side=side,
        dtype="int8-qksv/fp16-elsewhere" if precision == PRECISION_INT8_FP16 else dtype_name,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        lepe=lepe,
    )
    return config, attn, queries, keys, values


def make_workload(attn, queries, keys, values, device: torch.device) -> Callable[[], torch.Tensor]:
    @torch.no_grad()
    def run_once() -> torch.Tensor:
        return attn(queries, keys, values)

    return run_once


def trim_samples(samples: list[float], trim_fraction: float) -> list[float]:
    if not 0 <= trim_fraction < 0.5:
        raise SystemExit("--trim must be in [0, 0.5).")
    ordered = sorted(samples)
    trim = int(len(ordered) * trim_fraction)
    if trim == 0:
        return ordered
    kept = ordered[trim:-trim]
    if not kept:
        raise SystemExit("Trim settings removed all samples; increase --iters or reduce --trim.")
    return kept


def measure_latency(
    *,
    config: BenchConfig,
    run_once: Callable[[], torch.Tensor],
    warmup: int = WARMUP,
    iters: int = ITERS,
    trim_fraction: float = TRIM,
) -> LatencyResult:
    if iters <= 0:
        raise SystemExit("--iters must be positive.")
    device_type = "cuda" if is_cuda_backend(config.backend) else "cpu"
    for _ in range(warmup):
        run_once()
    if device_type == "cuda":
        torch.cuda.synchronize()

    samples: list[float] = []
    if device_type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timing_batch_size = (
            1
            if config.n == CUDA_UNBATCHED_N
            else min(CUDA_TIMING_BATCH_SIZE, iters)
        )
        complete_batches, remainder = divmod(iters, timing_batch_size)
        batch_sizes = [timing_batch_size] * complete_batches
        if remainder:
            batch_sizes.append(remainder)
        for batch_size in batch_sizes:
            starter.record()
            for _ in range(batch_size):
                run_once()
            ender.record()
            ender.synchronize()
            samples.append(float(starter.elapsed_time(ender)) / batch_size)
    else:
        timing_batch_size = 1
        for _ in range(iters):
            t0 = time.perf_counter()
            run_once()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)

    kept = trim_samples(samples, trim_fraction)
    kept_sorted = sorted(kept)
    mid = len(kept_sorted) // 2
    median = kept_sorted[mid] if len(kept_sorted) % 2 else 0.5 * (kept_sorted[mid - 1] + kept_sorted[mid])

    return LatencyResult(
        **asdict(config),
        warmup=warmup,
        iters=iters,
        timing_batch_size=timing_batch_size,
        timed_batches=len(samples),
        trim_fraction=trim_fraction,
        latency_ms_mean=sum(kept) / len(kept),
        latency_ms_median=median,
        latency_ms_min=min(kept),
        latency_ms_max=max(kept),
        samples_kept=len(kept),
    )


def run_power_loop(run_once: Callable[[], torch.Tensor], seconds: float, backend: str) -> int:
    deadline = time.perf_counter() + seconds
    iterations = 0
    while time.perf_counter() < deadline:
        run_once()
        iterations += 1
    if is_cuda_backend(backend):
        torch.cuda.synchronize()
    return iterations


def append_csv(path: Path, result: LatencyResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    if not write_header:
        with path.open(newline="") as fh:
            existing_header = next(csv.reader(fh), [])
        if existing_header != fieldnames:
            raise SystemExit(
                f"{path} uses an incompatible CSV schema; remove it and rerun."
            )
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_result(result: LatencyResult) -> None:
    print(
        "RESULT "
        f"backend={result.backend} n={result.n} k={result.k} levels={result.levels} "
        f"heads={result.heads} batch={result.batch} head_dim={result.head_dim} "
        f"precision={result.precision} dtype={result.dtype} "
        f"mean_ms={result.latency_ms_mean:.6f} median_ms={result.latency_ms_median:.6f} "
        f"timing_batch={result.timing_batch_size} "
        f"kept_timing_batches={result.samples_kept}/{result.timed_batches}"
    )


def run_latency_sweep(
    *,
    backend: str,
    n_values: Iterable[int],
    k_values: Iterable[int],
    out: Path,
    threads: int | None = None,
    heads: int = HEADS,
    head_dim: int = HEAD_DIM,
    batch: int = BATCH,
    precision: str = PRECISION_FP32,
    warmup: int = WARMUP,
    iters: int = ITERS,
    trim_fraction: float = TRIM,
) -> None:
    if threads is not None:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    if is_cuda_backend(backend):
        torch.backends.cudnn.benchmark = False

    for n in n_values:
        for k in k_values:
            config, attn, queries, keys, values = build_case(
                backend=backend,
                n=n,
                k=k,
                heads=heads,
                head_dim=head_dim,
                batch=batch,
                precision=precision,
            )
            run_once = make_workload(attn, queries, keys, values, get_device(backend))
            result = measure_latency(
                config=config,
                run_once=run_once,
                warmup=warmup,
                iters=iters,
                trim_fraction=trim_fraction,
            )
            print_result(result)
            append_csv(out, result)


def run_single_power_loop(
    *,
    backend: str,
    n: int,
    k: int,
    seconds: float = POWER_LOOP_SECONDS,
    threads: int | None = None,
    heads: int = HEADS,
    head_dim: int = HEAD_DIM,
    batch: int = BATCH,
    precision: str = PRECISION_FP32,
) -> None:
    if threads is not None:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    config, attn, queries, keys, values = build_case(
        backend=backend,
        n=n,
        k=k,
        heads=heads,
        head_dim=head_dim,
        batch=batch,
        precision=precision,
    )
    run_once = make_workload(attn, queries, keys, values, get_device(backend))
    iterations = run_power_loop(run_once, seconds, backend)
    print(
        f"POWER_LOOP backend={backend} n={n} k={k} heads={heads} batch={batch} precision={precision} "
        f"head_dim={head_dim} seconds={seconds:.1f} iterations={iterations}"
    )

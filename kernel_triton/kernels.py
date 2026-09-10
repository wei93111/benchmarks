"""Fused Triton inference kernels for QuadTree Attention-B.

The kernels intentionally target the benchmark configuration:

* FP32 forward inference
* arbitrary batch size
* head dimension 64
* 1 or 8 heads
* top-k in {4, 8, 16}

LePE and cross-level message accumulation remain in ``qt_bench.py`` so the
Triton and reference-CUDA backends execute identical surrounding operations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


SUPPORTED_HEADS = (1, 8)
SUPPORTED_TOPK = (4, 8, 16)
SUPPORTED_HEAD_DIM = 64


@triton.jit
def _coarse_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    topk_idx_ptr,
    n_tokens: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    n_channels: tl.constexpr,
    topk: tl.constexpr,
    sm_scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """Dense attention plus routing top-k, one program per (batch, query, head)."""
    pid = tl.program_id(0)
    head = pid % n_heads
    tmp = pid // n_heads
    query_idx = tmp % n_tokens
    batch = tmp // n_tokens

    d = tl.arange(0, block_d)
    d_mask = d < head_dim
    spatial_stride = n_tokens
    batch_stride = n_channels * spatial_stride
    channel = head * head_dim + d
    query_offsets = batch * batch_stride + channel * spatial_stride + query_idx
    query = tl.load(query_ptr + query_offsets, mask=d_mask, other=0.0).to(tl.float32)

    accumulator = tl.zeros((block_d,), dtype=tl.float32)
    running_max = -float("inf")
    running_sum = 0.0

    rank_offsets = tl.arange(0, topk)
    best_scores = tl.full((topk,), -float("inf"), dtype=tl.float32)
    best_indices = tl.zeros((topk,), dtype=tl.int32)
    for key_start in tl.range(0, n_tokens, block_n):
        key_offsets = key_start + tl.arange(0, block_n)
        key_mask = key_offsets < n_tokens

        key_ptrs = (
            key_ptr
            + batch * batch_stride
            + (head * head_dim + d[None, :]) * spatial_stride
            + key_offsets[:, None]
        )
        keys = tl.load(
            key_ptrs,
            mask=key_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * sm_scale
        scores = tl.where(key_mask, scores, -float("inf"))

        # Track the global top-k logits. Softmax is monotonic, so these are
        # exactly the same routing indices as top-k over probabilities.
        remaining_scores = scores
        for _ in tl.static_range(0, topk):
            local_position = tl.argmax(remaining_scores, axis=0)
            local_score = tl.max(remaining_scores, axis=0)
            local_index = key_start + local_position

            minimum_position = tl.argmin(best_scores, axis=0)
            minimum_score = tl.min(best_scores, axis=0)
            should_replace = local_score > minimum_score
            replace_mask = (rank_offsets == minimum_position) & should_replace
            best_scores = tl.where(replace_mask, local_score, best_scores)
            best_indices = tl.where(replace_mask, local_index, best_indices)
            remaining_scores = tl.where(
                tl.arange(0, block_n) == local_position,
                -float("inf"),
                remaining_scores,
            )

        # Online softmax and weighted-V accumulation.
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        probabilities = tl.where(key_mask, probabilities, 0.0)

        value_ptrs = (
            value_ptr
            + batch * batch_stride
            + (head * head_dim + d[None, :]) * spatial_stride
            + key_offsets[:, None]
        )
        values = tl.load(
            value_ptrs,
            mask=key_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        accumulator = accumulator * old_scale + tl.sum(
            probabilities[:, None] * values,
            axis=0,
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = new_max

    output_offsets = ((batch * n_tokens + query_idx) * n_heads + head) * head_dim + d
    tl.store(
        output_ptr + output_offsets,
        accumulator / running_sum,
        mask=d_mask,
    )

    topk_offsets = ((batch * n_tokens + query_idx) * topk + rank_offsets) * n_heads + head
    tl.store(topk_idx_ptr + topk_offsets, best_indices)


@triton.jit
def _fine_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    previous_topk_idx_ptr,
    output_ptr,
    next_topk_idx_ptr,
    height: tl.constexpr,
    width: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    n_channels: tl.constexpr,
    n_parent: tl.constexpr,
    previous_topk: tl.constexpr,
    next_topk: tl.constexpr,
    sm_scale: tl.constexpr,
    block_candidates: tl.constexpr,
    block_queries: tl.constexpr,
    block_d: tl.constexpr,
):
    """Sparse attention for four query children, fused in one program."""
    pid = tl.program_id(0)
    head = pid % n_heads
    tmp = pid // n_heads
    parent_idx = tmp % n_parent
    batch = tmp // n_parent

    parent_width = width // 2
    parent_y = parent_idx // parent_width
    parent_x = parent_idx - parent_y * parent_width
    query_in_parent = tl.arange(0, block_queries)
    query_mask = query_in_parent < 4
    query_y = parent_y * 2 + query_in_parent // 2
    query_x = parent_x * 2 + query_in_parent % 2
    query_spatial_idx = query_y * width + query_x

    d = tl.arange(0, block_d)
    d_mask = d < head_dim
    spatial_stride = height * width
    batch_stride = n_channels * spatial_stride
    query_offsets = (
        batch * batch_stride
        + (head * head_dim + d[None, :]) * spatial_stride
        + query_spatial_idx[:, None]
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=query_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    candidate_offset = tl.arange(0, block_candidates)
    candidate_count = previous_topk * 4
    candidate_mask = candidate_offset < candidate_count
    previous_rank = candidate_offset // 4
    child = candidate_offset % 4

    previous_index_offsets = (
        ((batch * n_parent + parent_idx) * previous_topk + previous_rank) * n_heads + head
    )
    previous_indices = tl.load(
        previous_topk_idx_ptr + previous_index_offsets,
        mask=candidate_mask,
        other=0,
    )
    previous_width = width // 2
    previous_y = previous_indices // previous_width
    previous_x = previous_indices - previous_y * previous_width
    candidate_y = previous_y * 2 + child // 2
    candidate_x = previous_x * 2 + child % 2
    candidate_indices = candidate_y * width + candidate_x

    key_ptrs = (
        key_ptr
        + batch * batch_stride
        + (head * head_dim + d[None, :]) * spatial_stride
        + candidate_indices[:, None]
    )
    keys = tl.load(
        key_ptrs,
        mask=candidate_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = (
        tl.dot(query, tl.trans(keys), input_precision="ieee") * sm_scale
    )
    scores = tl.where(
        query_mask[:, None] & candidate_mask[None, :],
        scores,
        -float("inf"),
    )

    row_max = tl.max(scores, axis=1)
    probabilities = tl.exp(scores - row_max[:, None])
    probabilities = tl.where(
        query_mask[:, None] & candidate_mask[None, :],
        probabilities,
        0.0,
    )
    probabilities /= tl.sum(probabilities, axis=1)[:, None]

    value_ptrs = (
        value_ptr
        + batch * batch_stride
        + (head * head_dim + d[None, :]) * spatial_stride
        + candidate_indices[:, None]
    )
    values = tl.load(
        value_ptrs,
        mask=candidate_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    message = tl.dot(probabilities, values, input_precision="ieee")

    output_offsets = (
        (((batch * n_parent + parent_idx) * 4 + query_in_parent[:, None]) * n_heads + head)
        * head_dim
        + d[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        message,
        mask=query_mask[:, None] & d_mask[None, :],
    )

    # Emit absolute linear indices in current-level spatial order. This removes
    # torch.topk, torch.gather, spatial rearrangement, and row/column stacking.
    remaining_scores = scores
    for rank in tl.static_range(0, next_topk):
        selected_position = tl.argmax(
            remaining_scores,
            axis=1,
            tie_break_left=True,
        )
        selected_index = tl.sum(
            tl.where(
                candidate_offset[None, :] == selected_position[:, None],
                candidate_indices[None, :],
                0,
            ),
            axis=1,
        )
        next_index_offset = (
            ((batch * height * width + query_spatial_idx) * next_topk + rank) * n_heads + head
        )
        tl.store(
            next_topk_idx_ptr + next_index_offset,
            selected_index,
            mask=query_mask,
        )
        remaining_scores = tl.where(
            candidate_offset[None, :] == selected_position[:, None],
            -float("inf"),
            remaining_scores,
        )


def _validate_common(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk: int,
) -> tuple[int, int, int, int, int]:
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("Triton attention requires CUDA tensors.")
    if query.dtype != torch.float32 or key.dtype != torch.float32 or value.dtype != torch.float32:
        raise TypeError("Triton baseline currently supports FP32 only.")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical NCHW shapes.")
    if query.ndim != 4:
        raise ValueError("query, key, and value must be four-dimensional NCHW tensors.")
    batch, channels, height, width = query.shape
    if batch < 1:
        raise ValueError(f"batch size must be positive, got {batch}.")
    if height != width:
        raise ValueError("Triton baseline requires square feature maps.")
    if topk not in SUPPORTED_TOPK:
        raise ValueError(f"topk must be one of {SUPPORTED_TOPK}, got {topk}.")
    if channels % SUPPORTED_HEAD_DIM != 0:
        raise ValueError(
            f"channels={channels} is not divisible by head_dim={SUPPORTED_HEAD_DIM}."
        )
    heads = channels // SUPPORTED_HEAD_DIM
    if heads not in SUPPORTED_HEADS:
        raise ValueError(f"heads must be one of {SUPPORTED_HEADS}, got {heads}.")
    return batch, channels, height, width, heads


def coarse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused dense coarse attention and return (message, topk_indices)."""
    batch, _, height, width, heads = _validate_common(query, key, value, topk)
    n_tokens = height * width
    if topk > n_tokens:
        raise ValueError(f"topk={topk} exceeds coarse token count {n_tokens}.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty(
        (batch, n_tokens, heads, SUPPORTED_HEAD_DIM),
        device=query.device,
        dtype=torch.float32,
    )
    topk_indices = torch.empty(
        (batch, n_tokens, topk, heads),
        device=query.device,
        dtype=torch.int64,
    )

    block_n = min(64, triton.next_power_of_2(n_tokens))
    grid = (batch * n_tokens * heads,)
    _coarse_attention_kernel[grid](
        query,
        key,
        value,
        output,
        topk_indices,
        n_tokens=n_tokens,
        n_heads=heads,
        head_dim=SUPPORTED_HEAD_DIM,
        n_channels=heads * SUPPORTED_HEAD_DIM,
        topk=topk,
        sm_scale=SUPPORTED_HEAD_DIM**-0.5,
        block_n=block_n,
        block_d=SUPPORTED_HEAD_DIM,
        num_warps=4,
        num_stages=2,
    )
    return output, topk_indices


def fine_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    previous_topk_indices: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one fused sparse fine level and return (message, next_topk_indices)."""
    batch, _, height, width, heads = _validate_common(query, key, value, topk)
    if height % 2 or width % 2:
        raise ValueError("Fine-level height and width must be divisible by 2.")
    if not previous_topk_indices.is_cuda:
        raise ValueError("previous_topk_indices must be a CUDA tensor.")
    if previous_topk_indices.dtype != torch.int64:
        raise TypeError("previous_topk_indices must have dtype torch.int64.")
    if previous_topk_indices.ndim != 4:
        raise ValueError("previous_topk_indices must have shape [B, N/4, K, H].")

    n_parent = (height // 2) * (width // 2)
    expected_prefix = (batch, n_parent)
    if previous_topk_indices.shape[:2] != expected_prefix:
        raise ValueError(
            "previous_topk_indices has incompatible parent count: "
            f"expected prefix {expected_prefix}, got {tuple(previous_topk_indices.shape[:2])}."
        )
    previous_topk = previous_topk_indices.shape[2]
    if previous_topk not in SUPPORTED_TOPK:
        raise ValueError(
            f"previous topk must be one of {SUPPORTED_TOPK}, got {previous_topk}."
        )
    if previous_topk_indices.shape[3] != heads:
        raise ValueError(
            f"previous indices have {previous_topk_indices.shape[3]} heads; expected {heads}."
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    previous_topk_indices = previous_topk_indices.contiguous()
    output = torch.empty(
        (batch, n_parent, 4, heads, SUPPORTED_HEAD_DIM),
        device=query.device,
        dtype=torch.float32,
    )
    next_topk_indices = torch.empty(
        (batch, height * width, topk, heads),
        device=query.device,
        dtype=torch.int64,
    )

    candidate_count = previous_topk * 4
    grid = (batch * n_parent * heads,)
    _fine_attention_kernel[grid](
        query,
        key,
        value,
        previous_topk_indices,
        output,
        next_topk_indices,
        height=height,
        width=width,
        n_heads=heads,
        head_dim=SUPPORTED_HEAD_DIM,
        n_channels=heads * SUPPORTED_HEAD_DIM,
        n_parent=n_parent,
        previous_topk=previous_topk,
        next_topk=topk,
        sm_scale=SUPPORTED_HEAD_DIM**-0.5,
        block_candidates=triton.next_power_of_2(candidate_count),
        block_queries=16,
        block_d=SUPPORTED_HEAD_DIM,
        num_warps=4,
        num_stages=2,
    )
    return output, next_topk_indices

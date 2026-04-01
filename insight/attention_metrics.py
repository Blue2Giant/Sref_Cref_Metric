#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attention metrics for step/block attention maps saved by
qwen_2511_attention_fullmap.py.

The expected payload is a torch.save dictionary with keys such as:
    - attn_tensor: [S, B, Q, K]
    - attn_mask: [S, B] optional
    - k_range_metadata_by_name: metadata for named K groups

All metric functions operate on tensors with shape [..., Q, K], so they can be
used on a full [S, B, Q, K] batch or on a single [Q, K] panel. If the input is
already on CUDA, all computations stay on CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


EPS = 1e-12
DEFAULT_GROUP_NAMES = ("text", "cref", "sref")
ATTN_TENSOR_KEYS = ("attn_tensor", "attention_step_block_grid", "attention", "attn")


@dataclass
class LoadedAttentionBatch:
    attn: torch.Tensor
    attn_mask: Optional[torch.Tensor]
    k_range_metadata_by_name: Dict[str, Dict[str, object]]
    payload: Dict[str, object]
    tensor_key: str


def _resolve_device(device: Optional[str | torch.device]) -> torch.device:
    if device is None or str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
    }
    key = str(dtype).strip().lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return table[key]


def _broadcast_mask(mask: torch.Tensor, target_ndim: int) -> torch.Tensor:
    out = mask
    while out.ndim < target_ndim:
        out = out.unsqueeze(-1)
    return out


def _nan_like(x: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, float("nan"))


def _apply_valid_mask(x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return x
    mask = valid_mask.to(device=x.device, dtype=torch.bool)
    if mask.shape != x.shape:
        raise ValueError(f"Mask shape mismatch: mask={tuple(mask.shape)} vs tensor={tuple(x.shape)}")
    return x.masked_fill(~mask, float("nan"))


def _sanitize_attention(attn: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    out = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    if attn_mask is not None:
        mask = _broadcast_mask(attn_mask.to(device=out.device, dtype=torch.bool), out.ndim)
        out = torch.where(mask, out, torch.zeros_like(out))
    return out


def _resolve_attn_tensor_from_payload(payload: Mapping[str, object]) -> Tuple[str, torch.Tensor]:
    for key in ATTN_TENSOR_KEYS:
        value = payload.get(key)
        if torch.is_tensor(value):
            return key, value
    raise KeyError(f"None of attention tensor keys were found: {ATTN_TENSOR_KEYS}")


def _build_range_metadata_from_ranges(
    ranges: Iterable[Tuple[str, int, int]],
    sample_indices: Sequence[int],
) -> Dict[str, Dict[str, object]]:
    sample_indices = [int(x) for x in sample_indices]
    out: Dict[str, Dict[str, object]] = {}
    for name, start, end in ranges:
        start = int(start)
        end = int(end)
        loc = [idx for idx, sample_idx in enumerate(sample_indices) if start <= sample_idx < end]
        item = {
            "name": str(name),
            "full_start": start,
            "full_end_exclusive": end,
            "full_end_inclusive": end - 1,
            "full_length": max(end - start, 0),
            "sample_start": None,
            "sample_end_exclusive": None,
            "sample_end_inclusive": None,
            "sample_length": 0,
        }
        if loc:
            sample_start = int(loc[0])
            sample_end_exclusive = int(loc[-1]) + 1
            item["sample_start"] = sample_start
            item["sample_end_exclusive"] = sample_end_exclusive
            item["sample_end_inclusive"] = sample_end_exclusive - 1
            item["sample_length"] = sample_end_exclusive - sample_start
        out[str(name)] = item
    return out


def _build_ref_ranges(total_len: int, num_refs: int) -> Sequence[Tuple[int, int]]:
    if num_refs <= 0:
        return []
    total_len = max(int(total_len), 0)
    base = total_len // num_refs
    rem = total_len % num_refs
    cursor = 0
    out = []
    for idx in range(num_refs):
        span = base + (1 if idx < rem else 0)
        end = cursor + span
        out.append((cursor, end))
        cursor = end
    return out


def _infer_k_semantic_ranges_from_payload(payload: Mapping[str, object]) -> Sequence[Tuple[str, int, int]]:
    k_full = int(payload.get("k_tokens_full", 0) or 0)
    if k_full <= 0:
        return []
    has_encoder = bool(payload.get("has_encoder", False))
    if not has_encoder:
        return [("noise", 0, k_full)]

    text_tokens_est = min(max(int(payload.get("text_tokens_est", 0) or 0), 0), k_full)
    ref_labels = [str(x) for x in (payload.get("ref_labels") or [])]
    ranges = []
    cursor = 0
    if text_tokens_est > 0:
        ranges.append(("text", 0, text_tokens_est))
        cursor = text_tokens_est
    remain = max(k_full - cursor, 0)
    if remain > 0 and ref_labels:
        for ridx, (start, end) in enumerate(_build_ref_ranges(remain, len(ref_labels))):
            name = ref_labels[ridx] if ridx < len(ref_labels) else f"ref{ridx}"
            ranges.append((name, cursor + start, cursor + end))
        cursor = k_full
    if cursor < k_full:
        ranges.append(("noise", cursor, k_full))
    return [(name, start, end) for name, start, end in ranges if end > start]


def resolve_k_range_metadata_by_name(payload: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    by_name = payload.get("k_range_metadata_by_name")
    if isinstance(by_name, dict) and by_name:
        return {str(k): dict(v) for k, v in by_name.items()}

    items = payload.get("k_range_metadata")
    if isinstance(items, list) and items:
        return {
            str(item.get("name")): dict(item)
            for item in items
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }

    ranges = payload.get("k_semantic_ranges")
    if isinstance(ranges, list) and ranges:
        sample_indices = [int(x) for x in (payload.get("k_sample_indices") or [])]
        norm_ranges = [(str(name), int(start), int(end)) for name, start, end in ranges]
        return _build_range_metadata_from_ranges(norm_ranges, sample_indices)

    inferred = _infer_k_semantic_ranges_from_payload(payload)
    if inferred:
        sample_indices = [int(x) for x in (payload.get("k_sample_indices") or [])]
        if not sample_indices:
            sample_indices = list(range(int(payload.get("k_tokens_full", 0) or 0)))
        return _build_range_metadata_from_ranges(inferred, sample_indices)

    raise KeyError("Failed to resolve k_range_metadata_by_name from payload")


def load_attention_batch(
    pt_path: str | Path,
    device: Optional[str | torch.device] = "auto",
    dtype: Optional[str | torch.dtype] = torch.float32,
    sanitize: bool = True,
) -> LoadedAttentionBatch:
    pt_path = Path(pt_path)
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)

    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {pt_path}, got {type(payload).__name__}")

    tensor_key, attn = _resolve_attn_tensor_from_payload(payload)
    if attn.ndim < 2:
        raise ValueError(f"Attention tensor must have at least 2 dims, got {tuple(attn.shape)}")
    attn = attn.to(device=resolved_device, dtype=resolved_dtype or attn.dtype)

    raw_mask = payload.get("attn_mask")
    attn_mask = raw_mask.to(device=resolved_device, dtype=torch.bool) if torch.is_tensor(raw_mask) else None

    if sanitize:
        attn = _sanitize_attention(attn, attn_mask)

    k_range_metadata_by_name = resolve_k_range_metadata_by_name(payload)
    return LoadedAttentionBatch(
        attn=attn,
        attn_mask=attn_mask,
        k_range_metadata_by_name=k_range_metadata_by_name,
        payload=dict(payload),
        tensor_key=tensor_key,
    )


def get_k_slice_from_meta(
    k_range_metadata_by_name: Mapping[str, Mapping[str, object]],
    group_name: str,
    use_sample: bool = True,
) -> Tuple[int, int]:
    if group_name not in k_range_metadata_by_name:
        raise KeyError(f"Group {group_name!r} not found in metadata. Available: {sorted(k_range_metadata_by_name)}")
    meta = k_range_metadata_by_name[group_name]
    if use_sample:
        start = meta.get("sample_start")
        end = meta.get("sample_end_exclusive")
        if start is None or end is None:
            raise ValueError(f"Group {group_name!r} has no sampled range metadata")
    else:
        start = meta.get("full_start")
        end = meta.get("full_end_exclusive")
        if start is None or end is None:
            raise ValueError(f"Group {group_name!r} has no full range metadata")
    return int(start), int(end)


def safe_normalize(x: torch.Tensor, dim: int, eps: float = EPS) -> torch.Tensor:
    denom = x.sum(dim=dim, keepdim=True).clamp_min(eps)
    return x / denom


def safe_entropy_from_prob(
    p: torch.Tensor,
    dim: int,
    normalized: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    p_safe = p.clamp_min(eps)
    ent = -(p * p_safe.log()).sum(dim=dim)
    if normalized:
        n = p.shape[dim]
        if n > 1:
            ent = ent / math.log(n)
        else:
            ent = torch.zeros_like(ent)
    return ent


def topk_mass_from_prob(p: torch.Tensor, k_top: int, dim: int) -> torch.Tensor:
    k_top = max(1, min(int(k_top), p.shape[dim]))
    values, _ = torch.topk(p, k=k_top, dim=dim)
    return values.sum(dim=dim)


def normalized_positions(length: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if length <= 1:
        return torch.zeros(length, device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype)


def _validate_group_slice(attn: torch.Tensor, k_start: int, k_end: int) -> Tuple[int, int]:
    k_start = int(k_start)
    k_end = int(k_end)
    k_total = int(attn.shape[-1])
    if k_start < 0 or k_end < 0 or k_start >= k_end or k_end > k_total:
        raise ValueError(f"Invalid K slice [{k_start}, {k_end}) for K={k_total}")
    return k_start, k_end


def k_marginal_prob_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    eps: float = EPS,
) -> torch.Tensor:
    k_start, k_end = _validate_group_slice(attn, k_start, k_end)
    group_attn = attn[..., :, k_start:k_end]
    k_mass = group_attn.sum(dim=-2)
    return safe_normalize(k_mass, dim=-1, eps=eps)


def mass_ratio_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    eps: float = EPS,
) -> torch.Tensor:
    k_start, k_end = _validate_group_slice(attn, k_start, k_end)
    group_mass = attn[..., :, k_start:k_end].sum(dim=(-1, -2))
    total_mass = attn.sum(dim=(-1, -2)).clamp_min(eps)
    return group_mass / total_mass


def enrichment_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    eps: float = EPS,
) -> torch.Tensor:
    k_start, k_end = _validate_group_slice(attn, k_start, k_end)
    k_total = max(int(attn.shape[-1]), 1)
    group_len = max(k_end - k_start, 1)
    ratio = mass_ratio_for_group(attn, k_start, k_end, eps=eps)
    uniform_expectation = group_len / k_total
    return ratio / max(uniform_expectation, eps)


def k_center_of_mass_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    eps: float = EPS,
) -> torch.Tensor:
    p_k = k_marginal_prob_for_group(attn, k_start, k_end, eps=eps)
    group_len = int(p_k.shape[-1])
    pos = normalized_positions(group_len, device=p_k.device, dtype=p_k.dtype)
    return (p_k * pos).sum(dim=-1)


def k_entropy_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    normalized: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    p_k = k_marginal_prob_for_group(attn, k_start, k_end, eps=eps)
    return safe_entropy_from_prob(p_k, dim=-1, normalized=normalized, eps=eps)


def k_topk_mass_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    topk: int = 5,
    eps: float = EPS,
) -> torch.Tensor:
    p_k = k_marginal_prob_for_group(attn, k_start, k_end, eps=eps)
    return topk_mass_from_prob(p_k, k_top=topk, dim=-1)


def q_response_for_group(attn: torch.Tensor, k_start: int, k_end: int) -> torch.Tensor:
    k_start, k_end = _validate_group_slice(attn, k_start, k_end)
    group_attn = attn[..., :, k_start:k_end]
    return group_attn.sum(dim=-1)


def q_marginal_prob_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    eps: float = EPS,
) -> torch.Tensor:
    q_resp = q_response_for_group(attn, k_start, k_end)
    return safe_normalize(q_resp, dim=-1, eps=eps)


def q_mean_variance_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_q = q_marginal_prob_for_group(attn, k_start, k_end, eps=eps)
    q_total = int(p_q.shape[-1])
    pos = normalized_positions(q_total, device=p_q.device, dtype=p_q.dtype)
    q_mean = (p_q * pos).sum(dim=-1)
    q_var = (p_q * (pos - q_mean.unsqueeze(-1)).pow(2)).sum(dim=-1)
    return q_mean, q_var


def q_entropy_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    normalized: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    p_q = q_marginal_prob_for_group(attn, k_start, k_end, eps=eps)
    return safe_entropy_from_prob(p_q, dim=-1, normalized=normalized, eps=eps)


def q_hhi_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    eps: float = EPS,
) -> torch.Tensor:
    del eps
    p_q = q_marginal_prob_for_group(attn, k_start, k_end, eps=EPS)
    return (p_q.pow(2)).sum(dim=-1)


def q_effective_count_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    eps: float = EPS,
) -> torch.Tensor:
    hhi = q_hhi_for_group(attn, k_start, k_end, eps=eps).clamp_min(eps)
    return 1.0 / hhi


def high_response_query_ratio_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    mode: str = "mean",
    alpha: float = 1.5,
    quantile: float = 0.9,
    absolute_threshold: Optional[float] = None,
    eps: float = EPS,
) -> torch.Tensor:
    del eps
    q_resp = q_response_for_group(attn, k_start, k_end)
    mode = str(mode).lower().strip()

    if mode == "mean":
        threshold = alpha * q_resp.mean(dim=-1, keepdim=True)
    elif mode == "quantile":
        threshold = torch.quantile(q_resp, q=float(quantile), dim=-1, keepdim=True)
    elif mode == "absolute":
        if absolute_threshold is None:
            raise ValueError("absolute_threshold must be provided when mode='absolute'")
        threshold = torch.full_like(q_resp[..., :1], float(absolute_threshold))
    else:
        raise ValueError(f"Unknown high-response mode: {mode}")

    active = (q_resp > threshold).to(dtype=q_resp.dtype)
    return active.mean(dim=-1)


def qk_mutual_information_for_group(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    normalized: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    k_start, k_end = _validate_group_slice(attn, k_start, k_end)
    group_attn = attn[..., :, k_start:k_end]
    group_mass = group_attn.sum(dim=(-1, -2), keepdim=True)
    p_joint = group_attn / group_mass.clamp_min(eps)
    p_q = p_joint.sum(dim=-1, keepdim=True)
    p_k = p_joint.sum(dim=-2, keepdim=True)

    ratio = p_joint / (p_q * p_k).clamp_min(eps)
    mi = (p_joint * ratio.clamp_min(eps).log()).sum(dim=(-1, -2))
    has_mass = group_mass.squeeze(-1).squeeze(-1) > 0
    mi = torch.where(has_mass, mi, torch.zeros_like(mi))

    if not normalized:
        return mi

    h_q = -(p_q * p_q.clamp_min(eps).log()).sum(dim=-2).squeeze(-1)
    h_k = -(p_k * p_k.clamp_min(eps).log()).sum(dim=-1).squeeze(-1)
    denom = torch.sqrt(h_q * h_k).clamp_min(eps)
    nmi = mi / denom
    return torch.where(has_mass, nmi, torch.zeros_like(nmi))


def compute_group_metrics(
    attn: torch.Tensor,
    k_start: int,
    k_end: int,
    topk: int = 5,
    high_resp_mode: str = "quantile",
    high_resp_alpha: float = 1.5,
    high_resp_quantile: float = 0.9,
    high_resp_absolute_threshold: Optional[float] = None,
    eps: float = EPS,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    q_mean, q_var = q_mean_variance_for_group(attn, k_start, k_end, eps=eps)

    metrics = {
        "mass_ratio": mass_ratio_for_group(attn, k_start, k_end, eps=eps),
        "enrichment": enrichment_for_group(attn, k_start, k_end, eps=eps),
        "k_center": k_center_of_mass_for_group(attn, k_start, k_end, eps=eps),
        "k_entropy": k_entropy_for_group(attn, k_start, k_end, normalized=True, eps=eps),
        "k_topk_mass": k_topk_mass_for_group(attn, k_start, k_end, topk=topk, eps=eps),
        "q_mean": q_mean,
        "q_var": q_var,
        "q_entropy": q_entropy_for_group(attn, k_start, k_end, normalized=True, eps=eps),
        "q_hhi": q_hhi_for_group(attn, k_start, k_end, eps=eps),
        "q_effective_count": q_effective_count_for_group(attn, k_start, k_end, eps=eps),
        "high_response_query_ratio": high_response_query_ratio_for_group(
            attn,
            k_start,
            k_end,
            mode=high_resp_mode,
            alpha=high_resp_alpha,
            quantile=high_resp_quantile,
            absolute_threshold=high_resp_absolute_threshold,
            eps=eps,
        ),
        "qk_mutual_information": qk_mutual_information_for_group(
            attn,
            k_start,
            k_end,
            normalized=False,
            eps=eps,
        ),
        "qk_normalized_mutual_information": qk_mutual_information_for_group(
            attn,
            k_start,
            k_end,
            normalized=True,
            eps=eps,
        ),
    }

    if valid_mask is not None:
        for key, value in metrics.items():
            metrics[key] = _apply_valid_mask(value, valid_mask)
    return metrics


def compute_metrics_for_named_groups(
    attn: torch.Tensor,
    k_range_metadata_by_name: Mapping[str, Mapping[str, object]],
    group_names: Sequence[str] = DEFAULT_GROUP_NAMES,
    use_sample: bool = True,
    topk: int = 5,
    high_resp_mode: str = "quantile",
    high_resp_alpha: float = 1.5,
    high_resp_quantile: float = 0.9,
    high_resp_absolute_threshold: Optional[float] = None,
    eps: float = EPS,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for group_name in group_names:
        k_start, k_end = get_k_slice_from_meta(
            k_range_metadata_by_name,
            group_name,
            use_sample=use_sample,
        )
        out[group_name] = compute_group_metrics(
            attn=attn,
            k_start=k_start,
            k_end=k_end,
            topk=topk,
            high_resp_mode=high_resp_mode,
            high_resp_alpha=high_resp_alpha,
            high_resp_quantile=high_resp_quantile,
            high_resp_absolute_threshold=high_resp_absolute_threshold,
            eps=eps,
            valid_mask=valid_mask,
        )
    return out


def compute_metrics_for_named_groups_from_pt(
    pt_path: str | Path,
    device: Optional[str | torch.device] = "auto",
    dtype: Optional[str | torch.dtype] = torch.float32,
    group_names: Sequence[str] = DEFAULT_GROUP_NAMES,
    use_sample: bool = True,
    topk: int = 5,
    high_resp_mode: str = "quantile",
    high_resp_alpha: float = 1.5,
    high_resp_quantile: float = 0.9,
    high_resp_absolute_threshold: Optional[float] = None,
    eps: float = EPS,
    sanitize: bool = True,
) -> Tuple[LoadedAttentionBatch, Dict[str, Dict[str, torch.Tensor]]]:
    batch = load_attention_batch(
        pt_path=pt_path,
        device=device,
        dtype=dtype,
        sanitize=sanitize,
    )
    metrics = compute_metrics_for_named_groups(
        attn=batch.attn,
        k_range_metadata_by_name=batch.k_range_metadata_by_name,
        group_names=group_names,
        use_sample=use_sample,
        topk=topk,
        high_resp_mode=high_resp_mode,
        high_resp_alpha=high_resp_alpha,
        high_resp_quantile=high_resp_quantile,
        high_resp_absolute_threshold=high_resp_absolute_threshold,
        eps=eps,
        valid_mask=batch.attn_mask,
    )
    return batch, metrics


def reduce_metric_tensor(
    metric: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    finite = torch.isfinite(metric)
    if not torch.any(finite):
        return torch.tensor(float("nan"), device=metric.device, dtype=metric.dtype)
    values = metric[finite]
    reduction = str(reduction).lower().strip()
    if reduction == "mean":
        return values.mean()
    if reduction == "max":
        return values.max()
    if reduction == "min":
        return values.min()
    if reduction == "median":
        return values.median()
    raise ValueError(f"Unsupported reduction: {reduction}")


def summarize_group_metrics(
    metrics: Mapping[str, Mapping[str, torch.Tensor]],
    reduction: str = "mean",
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for group_name, group_metrics in metrics.items():
        out[group_name] = {}
        for metric_name, value in group_metrics.items():
            out[group_name][metric_name] = float(reduce_metric_tensor(value, reduction=reduction).item())
    return out

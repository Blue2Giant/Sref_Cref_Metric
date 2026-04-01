#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare attention-map metrics across an arbitrary number of key cohorts,
with compatibility for Flux cache-manifest payloads.

This script extends the multi-cohort comparison flow to support both:
1. direct tensor payloads with `attn_tensor`
2. manifest payloads with `storage_format=step_block_cache_v1`

Example:
python /data/benchmark_metrics/insight/attention_metrics_compare_key_groups_multi_flux.py \
    --root-dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/flux-klein-9b-attn-fullmap \
    --output-dir /data/benchmark_metrics/logs/flux_attn_key_group_compare \
    --summary-csv /data/benchmark_metrics/logs/flux_attention_metrics_summary.csv

python /data/benchmark_metrics/insight/attention_metrics_compare_key_groups_multi_flux.py \
  --root-dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/flux-klein-9b-attn-fullmap-1-1 \
  --output-dir /data/benchmark_metrics/logs/flux_attn_key_group_compare-1-1 \
  --cohort-dir /data/benchmark_metrics/insight/key_folder/flux \
  --summary-csv /data/benchmark_metrics/logs/flux_attention_metrics_summary-1-1.csv \
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_hex

try:
    from insight.attention_metrics import (
        ATTN_TENSOR_KEYS,
        DEFAULT_GROUP_NAMES,
        LoadedAttentionBatch,
        _resolve_device,
        _resolve_dtype,
        _sanitize_attention,
        compute_metrics_for_named_groups,
        get_k_slice_from_meta,
        resolve_k_range_metadata_by_name,
        summarize_group_metrics,
    )
    from insight.attention_metrics_compare_key_groups import (
        CohortSpec,
        SampleMetrics,
        ensure_2d_metric,
        load_samples_from_long_csv,
        log,
        most_common_grid,
        nanmean_sem,
        read_keys,
        resolve_pt_path,
        sanitize_filename,
        stack_by_cohort,
        write_long_and_agg_csvs,
        write_sample_manifest,
    )
except ModuleNotFoundError:
    from attention_metrics import (
        ATTN_TENSOR_KEYS,
        DEFAULT_GROUP_NAMES,
        LoadedAttentionBatch,
        _resolve_device,
        _resolve_dtype,
        _sanitize_attention,
        compute_metrics_for_named_groups,
        get_k_slice_from_meta,
        resolve_k_range_metadata_by_name,
        summarize_group_metrics,
    )
    from attention_metrics_compare_key_groups import (
        CohortSpec,
        SampleMetrics,
        ensure_2d_metric,
        load_samples_from_long_csv,
        log,
        most_common_grid,
        nanmean_sem,
        read_keys,
        resolve_pt_path,
        sanitize_filename,
        stack_by_cohort,
        write_long_and_agg_csvs,
        write_sample_manifest,
    )


DEFAULT_METRIC_NAMES = (
    "mass_ratio",
    "enrichment",
    "k_center",
    "k_entropy",
    "k_topk_mass",
    "q_mean",
    "q_var",
    "q_entropy",
    "q_hhi",
    "q_effective_count",
    "high_response_query_ratio",
    "qk_mutual_information",
    "qk_normalized_mutual_information",
)
DEFAULT_COHORT_DIR = Path("/data/benchmark_metrics/insight/key_folder/flux")
PALETTE_NAMES = ("tab20", "tab20b", "tab20c")


def parse_args():
    parser = argparse.ArgumentParser("Compare attention metrics across arbitrary key cohorts for Flux-compatible payloads")
    parser.add_argument(
        "--root-dir",
        default="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/flux-klein-9b-attn-fullmap",
        help="Root directory that contains *_attn folders",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/benchmark_metrics/insight/output/flux_attn_key_group_compare",
        help="Directory for CSVs and plots",
    )
    parser.add_argument(
        "--cohort-dir",
        default=str(DEFAULT_COHORT_DIR),
        help="Directory containing cohort txt files; each *.txt becomes one cohort",
    )
    parser.add_argument(
        "--cohort-txt",
        action="append",
        default=[],
        help="Explicit cohort txt file. Repeatable. If omitted, all *.txt under --cohort-dir are used.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Full summary CSV path. Default: <output-dir>/attention_metrics_summary_all.csv",
    )
    parser.add_argument(
        "--selected-long-csv",
        default="",
        help="Per-sample long CSV path. Default: <output-dir>/selected_metrics_long.csv",
    )
    parser.add_argument(
        "--cohort-agg-csv",
        default="",
        help="Aggregated cohort CSV path. Default: <output-dir>/cohort_step_block_agg.csv",
    )
    parser.add_argument("--device", default="cpu", help='Compute device such as "cpu" or "cuda:0"')
    parser.add_argument("--dtype", default="float32", help='Compute dtype such as "float32" or "float16"')
    parser.add_argument("--groups", default="text,cref,sref", help="Comma-separated group names")
    parser.add_argument("--metrics", default="", help="Comma-separated metric names. Default: all")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--summary-reduction", choices=["mean", "median", "max", "min"], default="mean")
    parser.add_argument("--limit-summary", type=int, default=0, help="Only export the first N summary rows for debugging")
    parser.add_argument("--max-samples-per-cohort", type=int, default=0, help="Only keep the first N keys per cohort for debugging")
    parser.add_argument("--skip-summary-export", action="store_true", help="Skip the full summary CSV export")
    parser.add_argument(
        "--reuse-selected-long-csv",
        action="store_true",
        help="Skip PT loading and rebuild plots from selected_metrics_long.csv",
    )
    return parser.parse_args()


def normalize_cohort_name(stem: str) -> str:
    stem = str(stem).strip()
    if not stem:
        return "cohort"
    sanitized = []
    for ch in stem:
        if ch.isalnum():
            sanitized.append(ch.lower())
        else:
            sanitized.append("_")
    value = "".join(sanitized).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or "cohort"


def build_palette(count: int) -> List[str]:
    colors: List[str] = []
    for palette_name in PALETTE_NAMES:
        cmap = plt.get_cmap(palette_name)
        total = getattr(cmap, "N", 0) or 20
        for idx in range(total):
            colors.append(to_hex(cmap(idx / max(total - 1, 1))))
    if not colors:
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    return [colors[idx % len(colors)] for idx in range(count)]


def discover_cohort_files(args) -> List[Path]:
    explicit = [Path(item) for item in args.cohort_txt if str(item).strip()]
    if explicit:
        return explicit

    cohort_dir = Path(args.cohort_dir)
    candidate_dirs: List[Path] = []
    seen = set()
    for candidate in (cohort_dir, cohort_dir / "flux", cohort_dir.parent):
        candidate = Path(candidate)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidate_dirs.append(candidate)

    errors: List[str] = []
    for candidate in candidate_dirs:
        if not candidate.is_dir():
            errors.append(f"missing_dir:{candidate}")
            continue
        txt_files = sorted(path for path in candidate.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
        if not txt_files:
            errors.append(f"no_txt:{candidate}")
            continue
        flux_txt = [path for path in txt_files if path.stem.startswith("flux_")]
        if flux_txt:
            return flux_txt
        return txt_files

    raise RuntimeError(f"Failed to discover cohort txt files. Tried: {errors}")


def build_cohort_specs(paths: Sequence[Path]) -> List[CohortSpec]:
    if not paths:
        raise RuntimeError("No cohort txt files provided")

    palette = build_palette(len(paths))
    specs: List[CohortSpec] = []
    used_names: Dict[str, int] = {}
    for idx, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"cohort txt not found: {path}")
        base_name = normalize_cohort_name(path.stem)
        seq = used_names.get(base_name, 0) + 1
        used_names[base_name] = seq
        name = base_name if seq == 1 else f"{base_name}_{seq}"
        specs.append(
            CohortSpec(
                name=name,
                label=path.stem,
                key_file=path,
                color=palette[idx],
            )
        )
    return specs


def write_cohort_manifest(path: Path, cohort_specs: Sequence[CohortSpec]):
    rows = [
        {
            "cohort_name": item.name,
            "cohort_label": item.label,
            "key_file": str(item.key_file),
            "color": item.color,
        }
        for item in cohort_specs
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cohort_name", "cohort_label", "key_file", "color"])
        writer.writeheader()
        writer.writerows(rows)


def pair_specs(cohort_specs: Sequence[CohortSpec]) -> List[Tuple[CohortSpec, CohortSpec]]:
    return list(itertools.combinations(cohort_specs, 2))


def write_plot_notes(path: Path, cohort_specs: Sequence[CohortSpec]):
    pairs = [f"{left.label} - {right.label}" for left, right in pair_specs(cohort_specs)]
    pair_preview = "\n".join(f"- {item}" for item in pairs[:20])
    if len(pairs) > 20:
        pair_preview += f"\n- ... total={len(pairs)} pairs"

    text = f"""# Plot Guide

1. `first_last_block/*.png`
Shows how each metric evolves over time at the earliest and deepest sampled block.

2. `cohort_heatmaps/*.png`
Each panel is the cohort mean over `block x step`.

3. `difference_heatmaps/<metric>/*.png`
Each pair of cohorts is written as a separate heatmap figure.
Current cohort pairs:
{pair_preview if pair_preview else "- none"}

4. `block_profiles/*.png`
Aggregates steps into early/mid/late windows and plots metric vs block.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def iter_pt_files(root_dir: Path) -> List[Path]:
    return sorted(path for path in root_dir.rglob("attention_step_block_grid.pt") if path.is_file())


def _safe_int(value, default: int = -1) -> int:
    if value is None:
        return int(default)
    return int(value)


def _safe_name_from_parent(parent_name: str) -> str:
    if parent_name.endswith("_attn"):
        return parent_name[: -len("_attn")]
    return parent_name


def _add_group_range_columns(row: Dict[str, object], group_name: str, meta: Dict[str, object]):
    row[f"{group_name}_full_k_start"] = _safe_int(meta.get("full_start"))
    row[f"{group_name}_full_k_end_exclusive"] = _safe_int(meta.get("full_end_exclusive"))
    row[f"{group_name}_full_k_end_inclusive"] = _safe_int(meta.get("full_end_inclusive"))
    row[f"{group_name}_full_k_length"] = _safe_int(meta.get("full_length"), default=0)
    row[f"{group_name}_sampled_k_start"] = _safe_int(meta.get("sample_start"))
    row[f"{group_name}_sampled_k_end_exclusive"] = _safe_int(meta.get("sample_end_exclusive"))
    row[f"{group_name}_sampled_k_end_inclusive"] = _safe_int(meta.get("sample_end_inclusive"))
    row[f"{group_name}_sampled_k_length"] = _safe_int(meta.get("sample_length"), default=0)


def _add_group_metric_columns(row: Dict[str, object], group_name: str, summary: Dict[str, float]):
    for metric_name, value in summary.items():
        row[f"{group_name}_{metric_name}"] = float(value)


def collect_fieldnames(rows: Iterable[Dict[str, object]]) -> List[str]:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    return fieldnames


def _normalize_block_tensor_map(data: Dict[object, object], source_path: Path) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for key, value in data.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Step cache entry {source_path} contains non-tensor block value for key={key!r}")
        out[int(key)] = value
    return out


def _load_step_block_entry(entry: object, base_dir: Path) -> Dict[int, torch.Tensor]:
    if isinstance(entry, dict):
        return _normalize_block_tensor_map(entry, base_dir)
    if isinstance(entry, (str, Path)):
        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry_path = (base_dir / entry_path).resolve()
        data = torch.load(entry_path, map_location="cpu")
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in step cache {entry_path}, got {type(data).__name__}")
        return _normalize_block_tensor_map(data, entry_path)
    raise TypeError(f"Unsupported step-block entry type: {type(entry).__name__}")


def _collect_step_and_block_ids(step_block_maps: Dict[int, object], base_dir: Path) -> Tuple[List[int], List[int]]:
    if not step_block_maps:
        raise RuntimeError("No step-block attention entries found")
    steps = sorted(int(x) for x in step_block_maps.keys())
    block_ids = set()
    for step in steps:
        cur = _load_step_block_entry(step_block_maps[step], base_dir)
        for block_id in cur.keys():
            block_ids.add(int(block_id))
    ordered_block_ids = sorted(block_ids)
    if not ordered_block_ids:
        raise RuntimeError("No attention blocks found in step-block cache")
    return steps, ordered_block_ids


def pack_step_block_maps(step_block_maps: Dict[int, object], base_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    steps, block_ids = _collect_step_and_block_ids(step_block_maps, base_dir)

    q_target = 0
    k_target = 0
    for step in steps:
        cur_map = _load_step_block_entry(step_block_maps[step], base_dir)
        for block_id in block_ids:
            mat = cur_map.get(block_id)
            if mat is None:
                continue
            q_target = max(q_target, int(mat.shape[0]))
            k_target = max(k_target, int(mat.shape[1]))
    if q_target <= 0 or k_target <= 0:
        raise RuntimeError("Attention cache is empty after scanning all steps/blocks")

    tensor = torch.full((len(steps), len(block_ids), q_target, k_target), float("nan"), dtype=torch.float32)
    mask = torch.zeros((len(steps), len(block_ids)), dtype=torch.bool)
    for sidx, step in enumerate(steps):
        cur_map = _load_step_block_entry(step_block_maps[step], base_dir)
        for bidx, block_id in enumerate(block_ids):
            mat = cur_map.get(block_id)
            if mat is None:
                continue
            cur = mat.detach().float().cpu()
            if int(cur.shape[0]) != q_target or int(cur.shape[1]) != k_target:
                cur = torch.nn.functional.interpolate(
                    cur.unsqueeze(0).unsqueeze(0),
                    size=(q_target, k_target),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            tensor[sidx, bidx] = cur
            mask[sidx, bidx] = True
    return tensor, mask, steps, block_ids


def _materialize_manifest_payload(payload: Dict[str, object], pt_path: Path) -> Dict[str, object]:
    step_block_files = payload.get("step_block_files")
    if not isinstance(step_block_files, dict) or not step_block_files:
        return payload

    step_block_maps: Dict[int, object] = {}
    for step_key, file_ref in step_block_files.items():
        step_block_maps[int(step_key)] = file_ref

    tensor, mask, steps, block_ids = pack_step_block_maps(step_block_maps, pt_path.parent)
    merged = dict(payload)
    merged["attn_tensor"] = tensor
    merged["attn_mask"] = mask
    merged.setdefault("steps", [int(x) for x in steps])
    merged.setdefault("block_ids", [int(x) for x in block_ids])
    merged.setdefault("q_tokens_full", _infer_token_total(merged.get("q_ranges")))
    merged.setdefault("k_tokens_full", _infer_token_total(merged.get("k_ranges")))
    if "text_tokens_est" not in merged:
        try:
            text_meta = resolve_k_range_metadata_by_name(merged).get("text")
            merged["text_tokens_est"] = int(text_meta.get("full_length", 0)) if text_meta else 0
        except Exception:
            merged["text_tokens_est"] = 0
    merged.setdefault("has_encoder", bool("text" in (merged.get("k_range_metadata_by_name") or {})))
    return merged


def _infer_token_total(ranges: object) -> int:
    if not isinstance(ranges, list):
        return -1
    total = 0
    valid = False
    for item in ranges:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            start = int(item[1])
            end = int(item[2])
        except Exception:
            continue
        total += max(end - start, 0)
        valid = True
    return total if valid else -1


def _resolve_attn_tensor_from_payload_flexible(payload: Dict[str, object], pt_path: Path) -> Tuple[str, torch.Tensor, Dict[str, object]]:
    for key in ATTN_TENSOR_KEYS:
        value = payload.get(key)
        if torch.is_tensor(value):
            return key, value, payload

    if payload.get("storage_format") == "step_block_cache_v1" or payload.get("step_block_files"):
        merged = _materialize_manifest_payload(payload, pt_path)
        value = merged.get("attn_tensor")
        if torch.is_tensor(value):
            return "step_block_cache_v1", value, merged

    raise KeyError(f"None of attention tensor keys were found: {ATTN_TENSOR_KEYS}")


def load_attention_batch_flux_compatible(
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

    tensor_key, attn, payload = _resolve_attn_tensor_from_payload_flexible(dict(payload), pt_path)
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


def compute_metrics_for_named_groups_from_pt_flux(
    pt_path: str | Path,
    device: Optional[str | torch.device] = "auto",
    dtype: Optional[str | torch.dtype] = torch.float32,
    group_names: Sequence[str] = DEFAULT_GROUP_NAMES,
    use_sample: bool = True,
    topk: int = 5,
    sanitize: bool = True,
) -> Tuple[LoadedAttentionBatch, Dict[str, Dict[str, torch.Tensor]]]:
    batch = load_attention_batch_flux_compatible(
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
        valid_mask=batch.attn_mask,
    )
    return batch, metrics


def build_row_flux(
    pt_path: Path,
    group_names: Sequence[str],
    reduction: str,
    device: str,
    dtype: str,
    topk: int,
) -> Dict[str, object]:
    batch, metrics = compute_metrics_for_named_groups_from_pt_flux(
        pt_path=pt_path,
        device=device,
        dtype=dtype,
        group_names=group_names,
        use_sample=True,
        topk=topk,
    )
    summary = summarize_group_metrics(metrics, reduction=reduction)
    payload = dict(batch.payload)
    payload.setdefault("q_tokens_full", _infer_token_total(payload.get("q_ranges")))
    payload.setdefault("k_tokens_full", _infer_token_total(payload.get("k_ranges")))
    if "text_tokens_est" not in payload:
        text_meta = batch.k_range_metadata_by_name.get("text")
        payload["text_tokens_est"] = int(text_meta.get("full_length", 0)) if text_meta else 0
    parent_name = pt_path.parent.name

    row: Dict[str, object] = {
        "name": _safe_name_from_parent(parent_name),
        "dir_name": parent_name,
        "pt_path": str(pt_path),
        "tensor_key": batch.tensor_key,
        "num_steps": int(batch.attn.shape[0]) if batch.attn.ndim >= 4 else -1,
        "num_blocks": int(batch.attn.shape[1]) if batch.attn.ndim >= 4 else -1,
        "num_q": int(batch.attn.shape[-2]),
        "num_k": int(batch.attn.shape[-1]),
        "q_tokens_full": _safe_int(payload.get("q_tokens_full")),
        "k_tokens_full": _safe_int(payload.get("k_tokens_full")),
        "q_sample_len": len(payload.get("q_sample_indices") or []),
        "k_sample_len": len(payload.get("k_sample_indices") or []),
        "step_stride": _safe_int(payload.get("step_stride")),
        "block_stride": _safe_int(payload.get("block_stride")),
        "aggregate_head": str(payload.get("aggregate_head", "")),
        "has_encoder": int(bool(payload.get("has_encoder", False))),
        "text_tokens_est": _safe_int(payload.get("text_tokens_est"), default=0),
        "valid_panels": int(batch.attn_mask.sum().item()) if batch.attn_mask is not None else -1,
        "total_panels": int(batch.attn_mask.numel()) if batch.attn_mask is not None else -1,
    }

    for group_name in group_names:
        if group_name not in batch.k_range_metadata_by_name:
            continue
        meta = batch.k_range_metadata_by_name[group_name]
        _add_group_range_columns(row, group_name, meta)
        _add_group_metric_columns(row, group_name, summary[group_name])
        sampled_start, sampled_end = get_k_slice_from_meta(batch.k_range_metadata_by_name, group_name, use_sample=True)
        full_start, full_end = get_k_slice_from_meta(batch.k_range_metadata_by_name, group_name, use_sample=False)
        row[f"{group_name}_sampled_k_range"] = f"[{sampled_start},{sampled_end})"
        row[f"{group_name}_full_k_range"] = f"[{full_start},{full_end})"

    return row


def export_summary_csv_flux(
    *,
    root_dir: Path,
    out_csv: Path,
    group_names: Sequence[str],
    reduction: str,
    device: str,
    dtype: str,
    topk: int,
    limit: int,
):
    pt_files = iter_pt_files(root_dir)
    if limit > 0:
        pt_files = pt_files[:limit]
    if not pt_files:
        raise RuntimeError(f"No attention_step_block_grid.pt files found under {root_dir}")

    log(f"[summary] root_dir={root_dir}")
    log(f"[summary] out_csv={out_csv}")
    log(f"[summary] num_pt_files={len(pt_files)}")

    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for idx, pt_path in enumerate(pt_files, start=1):
        try:
            rows.append(
                build_row_flux(
                    pt_path=pt_path,
                    group_names=group_names,
                    reduction=reduction,
                    device=device,
                    dtype=dtype,
                    topk=topk,
                )
            )
            if idx % 20 == 0 or idx == len(pt_files):
                log(f"[summary] processed {idx}/{len(pt_files)}")
        except Exception as exc:
            failures.append(f"{pt_path}\t{type(exc).__name__}\t{exc}")
            log(f"[summary] fail {pt_path} :: {type(exc).__name__}: {exc}")

    if not rows:
        raise RuntimeError("Summary export failed for all files")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = collect_fieldnames(rows)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        fail_path = out_csv.with_suffix(out_csv.suffix + ".failures.txt")
        fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        log(f"[summary] failures={len(failures)} log={fail_path}")
    log(f"[summary] rows_written={len(rows)} csv_saved={out_csv}")


def collect_samples_flux(
    *,
    root_dir: Path,
    cohort_specs: Sequence[CohortSpec],
    group_names: Sequence[str],
    metric_names: Sequence[str],
    device: str,
    dtype: str,
    topk: int,
    max_samples_per_cohort: int,
    output_dir: Path,
) -> List[SampleMetrics]:
    selected: List[SampleMetrics] = []
    missing_manifest: List[Dict[str, object]] = []
    overlap_map: Dict[str, List[str]] = {}

    for cohort in cohort_specs:
        keys = read_keys(cohort.key_file)
        if max_samples_per_cohort > 0:
            keys = keys[:max_samples_per_cohort]
        log(f"[cohort] {cohort.label}: requested_keys={len(keys)} from {cohort.key_file}")
        for key in keys:
            overlap_map.setdefault(key, []).append(cohort.name)
            pt_path = resolve_pt_path(root_dir, key)
            if pt_path is None:
                missing_manifest.append(
                    {
                        "cohort_name": cohort.name,
                        "cohort_label": cohort.label,
                        "key": key,
                        "status": "missing_pt",
                    }
                )
                continue
            try:
                batch, metrics = compute_metrics_for_named_groups_from_pt_flux(
                    pt_path=pt_path,
                    device=device,
                    dtype=dtype,
                    group_names=group_names,
                    use_sample=True,
                    topk=topk,
                )
            except Exception as exc:
                missing_manifest.append(
                    {
                        "cohort_name": cohort.name,
                        "cohort_label": cohort.label,
                        "key": key,
                        "pt_path": str(pt_path),
                        "status": "metrics_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                log(f"[cohort] fail {key} :: {type(exc).__name__}: {exc}")
                continue

            step_stride = int(batch.payload.get("step_stride") or 1)
            block_stride = int(batch.payload.get("block_stride") or 1)
            if step_stride <= 0:
                step_stride = 1
            if block_stride <= 0:
                block_stride = 1

            if batch.payload.get("steps"):
                step_values = np.asarray([int(x) for x in batch.payload.get("steps")], dtype=np.int64)
            else:
                step_values = np.arange(int(batch.attn.shape[0]), dtype=np.int64) * step_stride
            if batch.payload.get("block_ids"):
                block_values = np.asarray([int(x) for x in batch.payload.get("block_ids")], dtype=np.int64)
            else:
                block_values = np.arange(int(batch.attn.shape[1]), dtype=np.int64) * block_stride

            filtered_metrics: Dict[str, Dict[str, np.ndarray]] = {}
            for group_name in group_names:
                if group_name not in metrics:
                    continue
                filtered_metrics[group_name] = {}
                for metric_name in metric_names:
                    if metric_name not in metrics[group_name]:
                        continue
                    arr = metrics[group_name][metric_name].detach().cpu().numpy()
                    filtered_metrics[group_name][metric_name] = ensure_2d_metric(
                        arr,
                        key=key,
                        group_name=group_name,
                        metric_name=metric_name,
                    )

            selected.append(
                SampleMetrics(
                    cohort_name=cohort.name,
                    cohort_label=cohort.label,
                    cohort_color=cohort.color,
                    key=key,
                    pt_path=pt_path,
                    step_values=step_values,
                    block_values=block_values,
                    step_stride=step_stride,
                    block_stride=block_stride,
                    metrics=filtered_metrics,
                )
            )

    overlaps = {k: v for k, v in overlap_map.items() if len(v) > 1}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "missing_keys.json").write_text(json.dumps(missing_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "overlap_keys.json").write_text(json.dumps(overlaps, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[cohort] resolved_samples={len(selected)} missing={len(missing_manifest)} overlaps={len(overlaps)}")
    return selected


def infer_time_windows(step_values: Sequence[int]) -> List[Tuple[str, np.ndarray]]:
    step_idx = np.arange(len(step_values))
    splits = np.array_split(step_idx, 3)
    labels = ("early", "mid", "late")
    out = []
    for label, indices in zip(labels, splits):
        if len(indices) == 0:
            continue
        out.append((label, indices))
    return out


def add_shared_legend(fig: plt.Figure, axes: np.ndarray):
    legend_items: Dict[str, object] = {}
    for ax in np.asarray(axes).reshape(-1):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label in legend_items:
                continue
            legend_items[label] = handle
    if not legend_items:
        return
    legend = fig.legend(
        list(legend_items.values()),
        list(legend_items.keys()),
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        frameon=True,
        borderaxespad=0.0,
        title="cohort",
    )
    frame = legend.get_frame()
    frame.set_edgecolor("0.82")
    frame.set_linewidth(0.8)


def plot_first_last_block(metric_name: str, group_names: Sequence[str], cohort_specs: Sequence[CohortSpec], samples: Sequence[SampleMetrics], out_path: Path):
    fig, axes = plt.subplots(len(group_names), 2, figsize=(14, 4 * len(group_names)), sharex=False, squeeze=False)
    for row, group_name in enumerate(group_names):
        step_grid, block_grid = most_common_grid(samples, group_name, metric_name)
        if step_grid is None or block_grid is None:
            continue
        first_idx = 0
        last_idx = len(block_grid) - 1
        step_vals = np.asarray(step_grid, dtype=np.int64)
        for col, block_idx in enumerate((first_idx, last_idx)):
            ax = axes[row][col]
            for cohort in cohort_specs:
                stack = stack_by_cohort(
                    samples,
                    cohort_name=cohort.name,
                    group_name=group_name,
                    metric_name=metric_name,
                    step_grid=step_grid,
                    block_grid=block_grid,
                )
                if stack is None:
                    continue
                line = stack[:, :, block_idx]
                mean, sem = nanmean_sem(line)
                ax.plot(step_vals, mean, color=cohort.color, linewidth=2, label=cohort.label)
                ax.fill_between(step_vals, mean - sem, mean + sem, color=cohort.color, alpha=0.18)
            block_label = "first_block" if block_idx == first_idx else "last_block"
            ax.set_title(f"{group_name} | {block_label}={block_grid[block_idx]}")
            ax.set_xlabel("time step")
            ax.set_ylabel(metric_name)
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    add_shared_legend(fig, axes)
    fig.suptitle(f"First/Last Block Time Curves | {metric_name}", y=0.985)
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_heatmaps(metric_name: str, group_names: Sequence[str], cohort_specs: Sequence[CohortSpec], samples: Sequence[SampleMetrics], out_path: Path):
    fig, axes = plt.subplots(len(group_names), len(cohort_specs), figsize=(4.8 * len(cohort_specs), 3.8 * len(group_names)), squeeze=False)
    for row, group_name in enumerate(group_names):
        step_grid, block_grid = most_common_grid(samples, group_name, metric_name)
        if step_grid is None or block_grid is None:
            continue
        mats = []
        means_by_cohort: Dict[str, np.ndarray] = {}
        for cohort in cohort_specs:
            stack = stack_by_cohort(
                samples,
                cohort_name=cohort.name,
                group_name=group_name,
                metric_name=metric_name,
                step_grid=step_grid,
                block_grid=block_grid,
            )
            if stack is None:
                continue
            mean = np.nanmean(stack, axis=0)
            means_by_cohort[cohort.name] = mean
            mats.append(mean)
        if not mats:
            continue
        vmin = min(float(np.nanmin(x)) for x in mats)
        vmax = max(float(np.nanmax(x)) for x in mats)
        images = []
        for col, cohort in enumerate(cohort_specs):
            ax = axes[row][col]
            mean = means_by_cohort.get(cohort.name)
            if mean is None:
                ax.set_axis_off()
                continue
            im = ax.imshow(
                mean.T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            images.append(im)
            ax.set_title(f"{group_name} | {cohort.label}")
            ax.set_xlabel("time step")
            ax.set_ylabel("block")
            ax.set_xticks(range(len(step_grid)))
            ax.set_xticklabels(step_grid, rotation=45, ha="right")
            ax.set_yticks(range(len(block_grid)))
            ax.set_yticklabels(block_grid)
        if images:
            fig.colorbar(images[0], ax=axes[row, :].tolist(), shrink=0.88, pad=0.01)
    fig.suptitle(f"Cohort Mean Heatmaps | {metric_name}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_difference_heatmaps(
    metric_name: str,
    group_names: Sequence[str],
    cohort_specs: Sequence[CohortSpec],
    samples: Sequence[SampleMetrics],
    out_dir: Path,
):
    pairs = pair_specs(cohort_specs)
    if not pairs:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for left, right in pairs:
        fig, axes = plt.subplots(len(group_names), 1, figsize=(6.4, 3.8 * len(group_names)), squeeze=False)
        plot_payloads: List[Tuple[int, Tuple[int, ...], Tuple[int, ...], np.ndarray]] = []
        extrema: List[float] = []

        for row, group_name in enumerate(group_names):
            step_grid, block_grid = most_common_grid(samples, group_name, metric_name)
            if step_grid is None or block_grid is None:
                continue
            left_stack = stack_by_cohort(
                samples,
                cohort_name=left.name,
                group_name=group_name,
                metric_name=metric_name,
                step_grid=step_grid,
                block_grid=block_grid,
            )
            right_stack = stack_by_cohort(
                samples,
                cohort_name=right.name,
                group_name=group_name,
                metric_name=metric_name,
                step_grid=step_grid,
                block_grid=block_grid,
            )
            if left_stack is None or right_stack is None:
                continue
            diff = np.nanmean(left_stack, axis=0) - np.nanmean(right_stack, axis=0)
            if not np.isfinite(diff).any():
                continue
            plot_payloads.append((row, step_grid, block_grid, diff))
            extrema.append(float(np.nanmax(np.abs(diff))))

        if not plot_payloads:
            plt.close(fig)
            continue

        vmax = max(extrema) if extrema else 0.0
        if not math.isfinite(vmax) or vmax <= 0:
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        image = None

        for row, step_grid, block_grid, diff in plot_payloads:
            ax = axes[row][0]
            image = ax.imshow(
                diff.T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="coolwarm",
                norm=norm,
            )
            ax.set_title(f"{group_names[row]} | {left.label} - {right.label}")
            ax.set_xlabel("time step")
            ax.set_ylabel("block")
            ax.set_xticks(range(len(step_grid)))
            ax.set_xticklabels(step_grid, rotation=45, ha="right")
            ax.set_yticks(range(len(block_grid)))
            ax.set_yticklabels(block_grid)

        for row, _group_name in enumerate(group_names):
            has_plot = any(item[0] == row for item in plot_payloads)
            if not has_plot:
                axes[row][0].set_axis_off()

        if image is not None:
            fig.colorbar(image, ax=axes[:, 0].tolist(), shrink=0.9, pad=0.02)

        fig.suptitle(f"Cohort Difference Heatmaps | {metric_name}", y=0.995)
        fig.subplots_adjust(left=0.1, right=0.88, top=0.93, bottom=0.08, hspace=0.35)
        pair_name = f"{sanitize_filename(left.label)}-minus-{sanitize_filename(right.label)}.png"
        fig.savefig(out_dir / pair_name, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_block_profiles(metric_name: str, group_names: Sequence[str], cohort_specs: Sequence[CohortSpec], samples: Sequence[SampleMetrics], out_path: Path):
    fig, axes = plt.subplots(len(group_names), 3, figsize=(15, 3.8 * len(group_names)), squeeze=False, sharex=False, sharey="row")
    for row, group_name in enumerate(group_names):
        step_grid, block_grid = most_common_grid(samples, group_name, metric_name)
        if step_grid is None or block_grid is None:
            continue
        windows = infer_time_windows(step_grid)
        block_vals = np.asarray(block_grid, dtype=np.int64)
        for col, (window_name, step_indices) in enumerate(windows):
            ax = axes[row][col]
            for cohort in cohort_specs:
                stack = stack_by_cohort(
                    samples,
                    cohort_name=cohort.name,
                    group_name=group_name,
                    metric_name=metric_name,
                    step_grid=step_grid,
                    block_grid=block_grid,
                )
                if stack is None:
                    continue
                prof = np.nanmean(stack[:, step_indices, :], axis=1)
                mean, sem = nanmean_sem(prof)
                ax.plot(block_vals, mean, color=cohort.color, linewidth=2, label=cohort.label)
                ax.fill_between(block_vals, mean - sem, mean + sem, color=cohort.color, alpha=0.18)
            step_values = [int(step_grid[i]) for i in step_indices]
            ax.set_title(f"{group_name} | {window_name}: {step_values}")
            ax.set_xlabel("block")
            ax.set_ylabel(metric_name)
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    add_shared_legend(fig, axes)
    fig.suptitle(f"Block Profiles in Early/Mid/Late Steps | {metric_name}", y=0.985)
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_names = tuple(x.strip() for x in str(args.groups).split(",") if x.strip()) or DEFAULT_GROUP_NAMES
    metric_names = tuple(x.strip() for x in str(args.metrics).split(",") if x.strip()) or DEFAULT_METRIC_NAMES

    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "attention_metrics_summary_all.csv"
    selected_long_csv = Path(args.selected_long_csv) if args.selected_long_csv else output_dir / "selected_metrics_long.csv"
    cohort_agg_csv = Path(args.cohort_agg_csv) if args.cohort_agg_csv else output_dir / "cohort_step_block_agg.csv"

    if not args.reuse_selected_long_csv and not root_dir.exists():
        raise FileNotFoundError(f"root-dir does not exist: {root_dir}")

    cohort_files = discover_cohort_files(args)
    cohort_specs = build_cohort_specs(cohort_files)
    write_cohort_manifest(output_dir / "cohort_manifest.csv", cohort_specs)
    (output_dir / "difference_pairs.json").write_text(
        json.dumps(
            [
                {
                    "left_name": left.name,
                    "left_label": left.label,
                    "right_name": right.name,
                    "right_label": right.label,
                }
                for left, right in pair_specs(cohort_specs)
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"[cohort] discovered_files={len(cohort_specs)} source_dir={args.cohort_dir} pairs={len(pair_specs(cohort_specs))}")

    if not args.skip_summary_export:
        export_summary_csv_flux(
            root_dir=root_dir,
            out_csv=summary_csv,
            group_names=group_names,
            reduction=str(args.summary_reduction),
            device=str(args.device),
            dtype=str(args.dtype),
            topk=int(args.topk),
            limit=int(args.limit_summary),
        )

    if args.reuse_selected_long_csv:
        if not selected_long_csv.is_file():
            raise FileNotFoundError(f"selected_long_csv not found: {selected_long_csv}")
        samples = load_samples_from_long_csv(
            long_csv=selected_long_csv,
            cohort_specs=cohort_specs,
            group_names=group_names,
            metric_names=metric_names,
        )
    else:
        samples = collect_samples_flux(
            root_dir=root_dir,
            cohort_specs=cohort_specs,
            group_names=group_names,
            metric_names=metric_names,
            device=str(args.device),
            dtype=str(args.dtype),
            topk=int(args.topk),
            max_samples_per_cohort=int(args.max_samples_per_cohort),
            output_dir=output_dir,
        )
    if not samples:
        raise RuntimeError("No samples resolved from the provided key files")

    write_sample_manifest(output_dir / "selected_sample_manifest.csv", samples)
    write_long_and_agg_csvs(
        samples=samples,
        group_names=group_names,
        metric_names=metric_names,
        long_csv=selected_long_csv,
        agg_csv=cohort_agg_csv,
    )
    write_plot_notes(output_dir / "plot_notes.md", cohort_specs)

    first_last_dir = output_dir / "plots" / "first_last_block"
    heatmap_dir = output_dir / "plots" / "cohort_heatmaps"
    diff_root_dir = output_dir / "plots" / "difference_heatmaps"
    block_profile_dir = output_dir / "plots" / "block_profiles"
    for metric_name in metric_names:
        safe = sanitize_filename(metric_name)
        log(f"[plot] metric={metric_name}")
        plot_first_last_block(metric_name, group_names, cohort_specs, samples, first_last_dir / f"{safe}.png")
        plot_cohort_heatmaps(metric_name, group_names, cohort_specs, samples, heatmap_dir / f"{safe}.png")
        plot_pairwise_difference_heatmaps(metric_name, group_names, cohort_specs, samples, diff_root_dir / safe)
        plot_block_profiles(metric_name, group_names, cohort_specs, samples, block_profile_dir / f"{safe}.png")

    meta = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "selected_long_csv": str(selected_long_csv),
        "cohort_agg_csv": str(cohort_agg_csv),
        "cohort_dir": str(args.cohort_dir),
        "group_names": list(group_names),
        "metric_names": list(metric_names),
        "num_selected_samples": len(samples),
        "num_cohorts": len(cohort_specs),
        "cohorts": [
            {
                "name": item.name,
                "label": item.label,
                "key_file": str(item.key_file),
                "color": item.color,
            }
            for item in cohort_specs
        ],
        "pairwise_differences": [
            {
                "left": left.label,
                "right": right.label,
            }
            for left, right in pair_specs(cohort_specs)
        ],
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"[done] output_dir={output_dir}")
    log(f"[done] summary_csv={summary_csv}")
    log(f"[done] selected_long_csv={selected_long_csv}")
    log(f"[done] cohort_agg_csv={cohort_agg_csv}")


if __name__ == "__main__":
    main()

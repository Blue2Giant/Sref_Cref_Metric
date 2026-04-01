#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare attention-map metrics across three cohorts of keys:
1. Success
2. Content leakage
3. Complete leakage

Outputs:
- A full summary CSV over all *_attn folders under root_dir
- A per-sample step/block long CSV for the selected keys
- A cohort-aggregated step/block CSV
- Diagnostic manifests for missing keys / overlaps
- Comparison plots that focus on:
  * first block vs last block over time
  * block-step heatmaps per cohort
  * pairwise cohort difference heatmaps
  * early/mid/late-step block profiles

python /data/benchmark_metrics/insight/attention_metrics_compare_key_groups.py \
    --root-dir /mnt/jfs/qwen-edit-attn-fullmap-keycolor-save-kfull \
    --output-dir /data/benchmark_metrics/logs/qwen_attn_key_group_compare \
    --success-txt /data/benchmark_metrics/insight/key_folder/success_key.txt \
    --content-leakage-txt /data/benchmark_metrics/insight/key_folder/content_leakage.txt \
    --complete-leakage-txt /data/benchmark_metrics/insight/key_folder/complet_leakage.txt \
    --summary-csv /data/benchmark_metrics/logs/attention_metrics_summary.csv

python /data/benchmark_metrics/insight/attention_metrics_compare_key_groups.py \
    --root-dir /mnt/jfs/qwen-edit-attn-fullmap-keycolor-save-kfull-1-1 \
    --output-dir /data/benchmark_metrics/logs/qwen_attn_key_group_compare-1-1 \
    --summary-csv /data/benchmark_metrics/logs/attention_metrics_summary.csv


python /data/benchmark_metrics/insight/attention_metrics_compare_key_groups_multi.py \
    --root-dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/flux-klein-9b-attn-fullmap \
    --output-dir /data/benchmark_metrics/logs/flux_attn_key_group_compare \
    --summary-csv /data/benchmark_metrics/logs/flux_attention_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    from insight.attention_metrics import compute_metrics_for_named_groups_from_pt
    from insight.attention_metrics_export_csv import build_row, collect_fieldnames, iter_pt_files
except ModuleNotFoundError:
    from attention_metrics import compute_metrics_for_named_groups_from_pt
    from attention_metrics_export_csv import build_row, collect_fieldnames, iter_pt_files


DEFAULT_GROUP_NAMES = ("text", "cref", "sref")
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


@dataclass(frozen=True)
class CohortSpec:
    name: str
    label: str
    key_file: Path
    color: str


@dataclass
class SampleMetrics:
    cohort_name: str
    cohort_label: str
    cohort_color: str
    key: str
    pt_path: Path
    step_values: np.ndarray
    block_values: np.ndarray
    step_stride: int
    block_stride: int
    metrics: Dict[str, Dict[str, np.ndarray]]


def parse_args():
    parser = argparse.ArgumentParser("Compare attention metrics across success/leakage key groups")
    parser.add_argument(
        "--root-dir",
        default="/mnt/jfs/qwen-edit-attn-fullmap-keycolor-save",
        help="Root directory that contains *_attn folders",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/benchmark_metrics/insight/output/qwen_attn_key_group_compare",
        help="Directory for CSVs and plots",
    )
    parser.add_argument(
        "--success-txt",
        default="/data/benchmark_metrics/insight/key_folder/success_key.txt",
        help="TXT file with success keys",
    )
    parser.add_argument(
        "--content-leakage-txt",
        default="/data/benchmark_metrics/insight/key_folder/content_leakage.txt",
        help="TXT file with content leakage keys",
    )
    parser.add_argument(
        "--complete-leakage-txt",
        default="/data/benchmark_metrics/insight/key_folder/complet_leakage.txt",
        help="TXT file with complete leakage keys",
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


def log(msg: str):
    print(msg, flush=True)


def sanitize_filename(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def read_keys(path: Path) -> List[str]:
    keys: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s or s.startswith("#"):
                continue
            if s in seen:
                continue
            seen.add(s)
            keys.append(s)
    return keys


def resolve_pt_path(root_dir: Path, key: str) -> Optional[Path]:
    key = str(key).strip()
    direct = root_dir / f"{key}_attn" / "attention_step_block_grid.pt"
    if direct.is_file():
        return direct

    parent = root_dir / f"{key}_attn"
    if parent.is_dir():
        alt = next(parent.glob("**/attention_step_block_grid.pt"), None)
        if alt is not None and alt.is_file():
            return alt

    # Fallback for keys with minor naming noise such as trailing spaces.
    pattern = f"{key}*_attn/attention_step_block_grid.pt"
    matches = sorted(root_dir.glob(pattern))
    if matches:
        return matches[0]
    return None


def ensure_2d_metric(arr: np.ndarray, *, key: str, group_name: str, metric_name: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=False)[:, None]
    if arr.ndim == 0:
        return arr.astype(np.float64, copy=False).reshape(1, 1)
    raise ValueError(f"Unexpected metric ndim={arr.ndim} for key={key} group={group_name} metric={metric_name}")


def export_summary_csv(
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
                build_row(
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


def collect_samples(
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
            batch, metrics = compute_metrics_for_named_groups_from_pt(
                pt_path=pt_path,
                device=device,
                dtype=dtype,
                group_names=group_names,
                use_sample=True,
                topk=topk,
            )
            step_stride = int(batch.payload.get("step_stride") or 1)
            block_stride = int(batch.payload.get("block_stride") or 1)
            if step_stride <= 0:
                step_stride = 1
            if block_stride <= 0:
                block_stride = 1

            step_values = np.arange(int(batch.attn.shape[0]), dtype=np.int64) * step_stride
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


def load_samples_from_long_csv(
    *,
    long_csv: Path,
    cohort_specs: Sequence[CohortSpec],
    group_names: Sequence[str],
    metric_names: Sequence[str],
) -> List[SampleMetrics]:
    color_by_cohort = {x.name: x.color for x in cohort_specs}
    default_label_by_cohort = {x.name: x.label for x in cohort_specs}
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}

    with long_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cohort_name = str(row.get("cohort_name", "")).strip()
            key = str(row.get("key", "")).strip()
            group_name = str(row.get("group_name", "")).strip()
            metric_name = str(row.get("metric_name", "")).strip()
            if not cohort_name or not key:
                continue
            if group_name not in group_names or metric_name not in metric_names:
                continue

            sample = grouped.setdefault(
                (cohort_name, key),
                {
                    "cohort_label": str(row.get("cohort_label", "")).strip() or default_label_by_cohort.get(cohort_name, cohort_name),
                    "step_map": {},
                    "block_map": {},
                    "values": {},
                },
            )
            step_idx = int(row["step_idx"])
            step_value = int(row["step_value"])
            block_idx = int(row["block_idx"])
            block_value = int(row["block_value"])
            raw_value = str(row.get("value", "")).strip()
            value = float(raw_value) if raw_value else math.nan

            step_map = sample["step_map"]
            block_map = sample["block_map"]
            values = sample["values"]
            step_map[step_idx] = step_value
            block_map[block_idx] = block_value
            values.setdefault(group_name, {}).setdefault(metric_name, {})[(step_idx, block_idx)] = value

    samples: List[SampleMetrics] = []
    for (cohort_name, key), payload in grouped.items():
        step_items = sorted(payload["step_map"].items())
        block_items = sorted(payload["block_map"].items())
        if not step_items or not block_items:
            continue

        step_values = np.asarray([value for _, value in step_items], dtype=np.int64)
        block_values = np.asarray([value for _, value in block_items], dtype=np.int64)
        step_pos = {idx: pos for pos, (idx, _) in enumerate(step_items)}
        block_pos = {idx: pos for pos, (idx, _) in enumerate(block_items)}

        metrics: Dict[str, Dict[str, np.ndarray]] = {}
        values = payload["values"]
        for group_name in group_names:
            group_metrics: Dict[str, np.ndarray] = {}
            for metric_name in metric_names:
                grid = np.full((len(step_values), len(block_values)), np.nan, dtype=np.float32)
                value_map = values.get(group_name, {}).get(metric_name, {})
                for (step_idx, block_idx), value in value_map.items():
                    grid[step_pos[step_idx], block_pos[block_idx]] = value
                group_metrics[metric_name] = grid
            metrics[group_name] = group_metrics

        step_stride = int(step_values[1] - step_values[0]) if len(step_values) > 1 else 1
        block_stride = int(block_values[1] - block_values[0]) if len(block_values) > 1 else 1
        samples.append(
            SampleMetrics(
                cohort_name=cohort_name,
                cohort_label=str(payload["cohort_label"]),
                cohort_color=color_by_cohort.get(cohort_name, "#666666"),
                key=key,
                pt_path=long_csv,
                step_values=step_values,
                block_values=block_values,
                step_stride=step_stride,
                block_stride=block_stride,
                metrics=metrics,
            )
        )

    log(f"[load] loaded {len(samples)} samples from {long_csv}")
    return samples


def write_sample_manifest(path: Path, samples: Sequence[SampleMetrics]):
    fieldnames = [
        "cohort_name",
        "cohort_label",
        "key",
        "pt_path",
        "num_steps",
        "num_blocks",
        "step_stride",
        "block_stride",
    ]
    rows = []
    for item in samples:
        rows.append(
            {
                "cohort_name": item.cohort_name,
                "cohort_label": item.cohort_label,
                "key": item.key,
                "pt_path": str(item.pt_path),
                "num_steps": len(item.step_values),
                "num_blocks": len(item.block_values),
                "step_stride": item.step_stride,
                "block_stride": item.block_stride,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_long_and_agg_csvs(
    *,
    samples: Sequence[SampleMetrics],
    group_names: Sequence[str],
    metric_names: Sequence[str],
    long_csv: Path,
    agg_csv: Path,
):
    long_fieldnames = [
        "cohort_name",
        "cohort_label",
        "key",
        "group_name",
        "metric_name",
        "step_idx",
        "step_value",
        "block_idx",
        "block_value",
        "value",
    ]
    agg_fieldnames = [
        "cohort_name",
        "cohort_label",
        "group_name",
        "metric_name",
        "step_idx",
        "step_value",
        "block_idx",
        "block_value",
        "n",
        "mean",
        "std",
        "sem",
        "median",
        "min",
        "max",
    ]

    bucket: Dict[Tuple[object, ...], List[float]] = {}
    long_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_csv.parent.mkdir(parents=True, exist_ok=True)

    with long_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_fieldnames)
        writer.writeheader()
        for item in samples:
            for group_name in group_names:
                group_metrics = item.metrics.get(group_name, {})
                for metric_name in metric_names:
                    mat = group_metrics.get(metric_name)
                    if mat is None:
                        continue
                    for step_idx in range(mat.shape[0]):
                        for block_idx in range(mat.shape[1]):
                            value = float(mat[step_idx, block_idx])
                            if not math.isfinite(value):
                                continue
                            row = {
                                "cohort_name": item.cohort_name,
                                "cohort_label": item.cohort_label,
                                "key": item.key,
                                "group_name": group_name,
                                "metric_name": metric_name,
                                "step_idx": int(step_idx),
                                "step_value": int(item.step_values[step_idx]),
                                "block_idx": int(block_idx),
                                "block_value": int(item.block_values[block_idx]),
                                "value": value,
                            }
                            writer.writerow(row)
                            key = (
                                item.cohort_name,
                                item.cohort_label,
                                group_name,
                                metric_name,
                                int(step_idx),
                                int(item.step_values[step_idx]),
                                int(block_idx),
                                int(item.block_values[block_idx]),
                            )
                            bucket.setdefault(key, []).append(value)

    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        for key in sorted(bucket.keys()):
            values = np.asarray(bucket[key], dtype=np.float64)
            n = int(values.size)
            std = float(np.std(values))
            sem = float(std / math.sqrt(n)) if n > 0 else float("nan")
            writer.writerow(
                {
                    "cohort_name": key[0],
                    "cohort_label": key[1],
                    "group_name": key[2],
                    "metric_name": key[3],
                    "step_idx": key[4],
                    "step_value": key[5],
                    "block_idx": key[6],
                    "block_value": key[7],
                    "n": n,
                    "mean": float(np.mean(values)),
                    "std": std,
                    "sem": sem,
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )


def most_common_grid(samples: Sequence[SampleMetrics], group_name: str, metric_name: str) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    counts: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
    for item in samples:
        mat = item.metrics.get(group_name, {}).get(metric_name)
        if mat is None:
            continue
        key = (tuple(int(x) for x in item.step_values.tolist()), tuple(int(x) for x in item.block_values.tolist()))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None, None
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return best[0], best[1]


def stack_by_cohort(
    samples: Sequence[SampleMetrics],
    *,
    cohort_name: str,
    group_name: str,
    metric_name: str,
    step_grid: Tuple[int, ...],
    block_grid: Tuple[int, ...],
) -> Optional[np.ndarray]:
    mats: List[np.ndarray] = []
    for item in samples:
        if item.cohort_name != cohort_name:
            continue
        if tuple(int(x) for x in item.step_values.tolist()) != step_grid:
            continue
        if tuple(int(x) for x in item.block_values.tolist()) != block_grid:
            continue
        mat = item.metrics.get(group_name, {}).get(metric_name)
        if mat is None:
            continue
        mats.append(mat.astype(np.float64, copy=False))
    if not mats:
        return None
    return np.stack(mats, axis=0)


def nanmean_sem(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(stack, axis=0)
    count = np.sum(np.isfinite(stack), axis=0)
    std = np.nanstd(stack, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = std / np.sqrt(np.maximum(count, 1))
    sem = np.where(count > 0, sem, np.nan)
    return mean, sem


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
            # The metric grid is indexed as [step, block], while imshow expects [y, x].
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


def plot_difference_heatmaps(metric_name: str, group_names: Sequence[str], samples: Sequence[SampleMetrics], out_path: Path):
    diff_specs = [
        ("content_leakage", "success", "content-success"),
        ("complete_leakage", "success", "complete-success"),
        ("complete_leakage", "content_leakage", "complete-content"),
    ]
    fig, axes = plt.subplots(len(group_names), len(diff_specs), figsize=(5.2 * len(diff_specs), 3.8 * len(group_names)), squeeze=False)
    for row, group_name in enumerate(group_names):
        step_grid, block_grid = most_common_grid(samples, group_name, metric_name)
        if step_grid is None or block_grid is None:
            continue
        diffs = []
        diff_values: List[Tuple[str, np.ndarray]] = []
        for lhs, rhs, label in diff_specs:
            left = stack_by_cohort(samples, cohort_name=lhs, group_name=group_name, metric_name=metric_name, step_grid=step_grid, block_grid=block_grid)
            right = stack_by_cohort(samples, cohort_name=rhs, group_name=group_name, metric_name=metric_name, step_grid=step_grid, block_grid=block_grid)
            if left is None or right is None:
                continue
            diff = np.nanmean(left, axis=0) - np.nanmean(right, axis=0)
            diff_values.append((label, diff))
            diffs.append(diff)
        if not diffs:
            continue
        vmax = max(abs(float(np.nanmin(x))) for x in diffs)
        vmax = max(vmax, max(abs(float(np.nanmax(x))) for x in diffs))
        if vmax <= 0:
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        images = []
        for col, spec in enumerate(diff_specs):
            ax = axes[row][col]
            label = spec[2]
            found = next((x for x in diff_values if x[0] == label), None)
            if found is None:
                ax.set_axis_off()
                continue
            # The diff grid is indexed as [step, block], while imshow expects [y, x].
            im = ax.imshow(
                found[1].T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="coolwarm",
                norm=norm,
            )
            images.append(im)
            ax.set_title(f"{group_name} | {label}")
            ax.set_xlabel("time step")
            ax.set_ylabel("block")
            ax.set_xticks(range(len(step_grid)))
            ax.set_xticklabels(step_grid, rotation=45, ha="right")
            ax.set_yticks(range(len(block_grid)))
            ax.set_yticklabels(block_grid)
        if images:
            fig.colorbar(images[0], ax=axes[row, :].tolist(), shrink=0.88, pad=0.01)
    fig.suptitle(f"Cohort Difference Heatmaps | {metric_name}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
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


def write_plot_notes(path: Path):
    text = """# Plot Guide

1. `first_last_block/*.png`
Shows how each metric evolves over time at the earliest and deepest sampled block.
Use it to answer:
- Does leakage emerge early or only after deep blocks?
- Is success characterized by delayed or sustained style focus?

2. `cohort_heatmaps/*.png`
Each panel is the cohort mean over `block x step`.
Use it to see where attention mass or dispersion concentrates jointly in time and depth.

3. `difference_heatmaps/*.png`
Shows direct deltas between cohorts with a zero-centered colormap.
Use it to see whether leakage differs from success at shallow blocks, late steps, or both.

4. `block_profiles/*.png`
Aggregates steps into early/mid/late windows and plots metric vs block.
Use it to judge whether leakage is mainly a shallow-layer or deep-layer phenomenon at each denoising stage.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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

    cohort_specs = [
        CohortSpec(
            name="success",
            label="Success",
            key_file=Path(args.success_txt),
            color="#2E8B57",
        ),
        CohortSpec(
            name="content_leakage",
            label="Content Leakage",
            key_file=Path(args.content_leakage_txt),
            color="#E69F00",
        ),
        CohortSpec(
            name="complete_leakage",
            label="Complete Leakage",
            key_file=Path(args.complete_leakage_txt),
            color="#D55E00",
        ),
    ]

    if not args.skip_summary_export:
        export_summary_csv(
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
        samples = collect_samples(
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
    write_plot_notes(output_dir / "plot_notes.md")

    first_last_dir = output_dir / "plots" / "first_last_block"
    heatmap_dir = output_dir / "plots" / "cohort_heatmaps"
    diff_dir = output_dir / "plots" / "difference_heatmaps"
    block_profile_dir = output_dir / "plots" / "block_profiles"
    for metric_name in metric_names:
        safe = sanitize_filename(metric_name)
        log(f"[plot] metric={metric_name}")
        plot_first_last_block(metric_name, group_names, cohort_specs, samples, first_last_dir / f"{safe}.png")
        plot_cohort_heatmaps(metric_name, group_names, cohort_specs, samples, heatmap_dir / f"{safe}.png")
        plot_difference_heatmaps(metric_name, group_names, samples, diff_dir / f"{safe}.png")
        plot_block_profiles(metric_name, group_names, cohort_specs, samples, block_profile_dir / f"{safe}.png")

    meta = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "selected_long_csv": str(selected_long_csv),
        "cohort_agg_csv": str(cohort_agg_csv),
        "group_names": list(group_names),
        "metric_names": list(metric_names),
        "num_selected_samples": len(samples),
        "cohorts": [
            {
                "name": x.name,
                "label": x.label,
                "key_file": str(x.key_file),
                "color": x.color,
            }
            for x in cohort_specs
        ],
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"[done] output_dir={output_dir}")
    log(f"[done] summary_csv={summary_csv}")
    log(f"[done] selected_long_csv={selected_long_csv}")
    log(f"[done] cohort_agg_csv={cohort_agg_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate compact paper-facing novelty figures for Qwen attention analysis.

Outputs:
1. Attention strips:
   - pick one representative key from each cohort txt
   - crop the first-block column from an existing attention grid PNG
   - transpose rows(steps) into a single horizontal strip
   - save both PNG and SVG

2. Metric focus figure:
   - reuse selected_metrics_long.csv
   - compare `sref` / `enrichment` across cohorts
   - only plot the first block
   - save both PNG and SVG

The script prefers the latest kfull/q-latent-range grid root when available,
but can fall back to any earlier local root that still contains the per-key
attention grid PNGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from insight.attention_metrics_compare_key_groups import (
        CohortSpec,
        load_samples_from_long_csv,
        log,
        most_common_grid,
        nanmean_sem,
        sanitize_filename,
        stack_by_cohort,
    )
except ModuleNotFoundError:
    from attention_metrics_compare_key_groups import (
        CohortSpec,
        load_samples_from_long_csv,
        log,
        most_common_grid,
        nanmean_sem,
        sanitize_filename,
        stack_by_cohort,
    )


DEFAULT_SELECTED_LONG_CSV = Path(
    "/data/benchmark_metrics/logs/qwen_attn_key_group_compare_kfull_1_1_q_latent_range_20260408_remote_root/selected_metrics_long.csv"
)
DEFAULT_OUTPUT_DIR = Path("/data/benchmark_metrics/logs/qwen_attention_paper_novelty_qwen_20260412_tmux")
DEFAULT_GRID_ROOTS = (
    Path("/mnt/jfs/qwen-edit-attn-fullmap-keycolor-save-kfull-1-1-q-latent-range"),
    Path("/data/benchmark_metrics/logs/qwen-edit-attn-fullmap-keycolor"),
)
DEFAULT_KEY_DIR = Path("/data/benchmark_metrics/insight/key_folder/qwen")
DEFAULT_EXPORT_FORMATS = ("png", "svg")


def parse_args():
    parser = argparse.ArgumentParser("Generate Qwen attention paper novelty figures")
    parser.add_argument(
        "--selected-long-csv",
        default=str(DEFAULT_SELECTED_LONG_CSV),
        help="selected_metrics_long.csv produced by attention_metrics_compare_key_groups.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for paper-facing outputs",
    )
    parser.add_argument(
        "--grid-root",
        action="append",
        default=[],
        help="Root directory that contains <key>_attn/attention_step_block_grid.png. Repeatable.",
    )
    parser.add_argument(
        "--success-txt",
        default=str(DEFAULT_KEY_DIR / "success.txt"),
        help="TXT file with success keys",
    )
    parser.add_argument(
        "--content-leakage-txt",
        default=str(DEFAULT_KEY_DIR / "content_leakage.txt"),
        help="TXT file with content leakage keys",
    )
    parser.add_argument(
        "--complete-leakage-txt",
        default=str(DEFAULT_KEY_DIR / "complet_leakage.txt"),
        help="TXT file with complete leakage keys",
    )
    parser.add_argument(
        "--metric-group",
        default="sref",
        help="Named attention group for the focus metric plot",
    )
    parser.add_argument(
        "--metric-name",
        default="enrichment",
        help="Metric to visualize for the focus metric plot",
    )
    parser.add_argument(
        "--focus-block-idx",
        type=int,
        default=0,
        help="0-based block index for the focus metric plot",
    )
    parser.add_argument(
        "--export-format",
        action="append",
        default=[],
        help="Output figure format. Repeatable. Default: png + svg",
    )
    parser.add_argument(
        "--step-label-mode",
        choices=["none", "index", "value"],
        default="value",
        help="How to label columns in the attention strips",
    )
    parser.add_argument(
        "--keys-per-cohort",
        type=int,
        default=4,
        help="How many keys to visualize for each cohort in the attention-strip figure",
    )
    parser.add_argument(
        "--step-sample-count",
        type=int,
        default=8,
        help="How many denoising steps to keep in the attention-strip figure",
    )
    parser.add_argument(
        "--strip-gap-px",
        type=int,
        default=34,
        help="White separator width between step columns in the attention strip",
    )
    parser.add_argument(
        "--strip-row-gap-px",
        type=int,
        default=24,
        help="White separator height between different keys in the attention-strip figure",
    )
    parser.add_argument(
        "--strip-border-px",
        type=int,
        default=8,
        help="White border width around each cropped panel",
    )
    parser.add_argument(
        "--strip-upscale",
        type=int,
        default=3,
        help="Nearest-neighbor upscale factor for each cropped attention panel",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=360,
        help="DPI for exported PNG figures",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.3,
        help="Line width for the focus metric plot",
    )
    return parser.parse_args()


def read_keys(path: Path) -> List[str]:
    keys: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = str(line or "").strip()
            if not value or value.startswith("#") or value in seen:
                continue
            seen.add(value)
            keys.append(value)
    return keys


def build_qwen_cohort_specs(args) -> List[CohortSpec]:
    return [
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


def normalize_formats(values: Sequence[str]) -> Tuple[str, ...]:
    items = [str(x).strip().lower() for x in values if str(x).strip()]
    if not items:
        items = list(DEFAULT_EXPORT_FORMATS)
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def resolve_grid_roots(args) -> List[Path]:
    roots = [Path(x) for x in args.grid_root if str(x).strip()]
    if not roots:
        roots = [Path(x) for x in DEFAULT_GRID_ROOTS]
    return roots


def resolve_grid_png(grid_roots: Sequence[Path], key: str) -> Optional[Path]:
    for root in grid_roots:
        candidate = root / f"{key}_attn" / "attention_step_block_grid.png"
        if candidate.is_file():
            return candidate
    return None


def select_keys_for_cohort(
    grid_roots: Sequence[Path],
    cohort: CohortSpec,
    *,
    max_keys: int,
) -> Tuple[List[Tuple[str, Path]], List[str]]:
    selected: List[Tuple[str, Path]] = []
    tried: List[str] = []
    for key in read_keys(cohort.key_file):
        tried.append(key)
        grid_png = resolve_grid_png(grid_roots, key)
        if grid_png is None:
            continue
        selected.append((key, grid_png))
        if len(selected) >= max(int(max_keys), 1):
            break
    return selected, tried


def _group_contiguous(indices: np.ndarray) -> List[Tuple[int, int]]:
    if indices.size <= 0:
        return []
    out: List[Tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for item in indices[1:]:
        value = int(item)
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = value
        prev = value
    out.append((start, prev))
    return out


def detect_panel_boundaries(
    arr_rgb: np.ndarray,
    *,
    pixel_threshold: int = 20,
    line_ratio_threshold: float = 0.7,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    gray = arr_rgb.mean(axis=2)
    dark = (gray < float(pixel_threshold)).astype(np.float32)
    row_score = dark.mean(axis=1)
    col_score = dark.mean(axis=0)

    row_groups = _group_contiguous(np.where(row_score >= float(line_ratio_threshold))[0])
    col_groups = _group_contiguous(np.where(col_score >= float(line_ratio_threshold))[0])
    if len(row_groups) < 2 or len(col_groups) < 2:
        raise RuntimeError(
            "Failed to detect subplot divider lines from attention grid PNG "
            f"(rows={len(row_groups)}, cols={len(col_groups)})"
        )
    return row_groups, col_groups


def extract_panel_boxes(boundary_groups: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    boxes: List[Tuple[int, int]] = []
    for idx in range(len(boundary_groups) - 1):
        start = int(boundary_groups[idx][1]) + 1
        end = int(boundary_groups[idx + 1][0]) - 1
        if end < start:
            continue
        boxes.append((start, end))
    return boxes


def step_labels_for_strip(
    count: int,
    *,
    mode: str,
    step_values: Optional[Sequence[int]] = None,
) -> List[str]:
    if mode == "none":
        return [""] * count
    if mode == "value" and step_values is not None and len(step_values) == count:
        return [str(int(x)) for x in step_values]
    return [str(idx) for idx in range(count)]


def sample_indices_evenly(total: int, target_count: int) -> List[int]:
    total = max(int(total), 0)
    target_count = max(int(target_count), 1)
    if total <= target_count:
        return list(range(total))
    idx = np.linspace(0, total - 1, num=target_count)
    idx = np.unique(np.round(idx).astype(np.int64))
    return [int(x) for x in idx.tolist()]


def _upscale_tile(tile: np.ndarray, scale: int) -> np.ndarray:
    scale = max(int(scale), 1)
    if scale <= 1:
        return tile
    return np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)


def extract_first_block_tiles(
    grid_png: Path,
    *,
    border_px: int,
    upscale: int,
    selected_step_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], int]:
    img = Image.open(grid_png).convert("RGB")
    arr = np.asarray(img)
    row_groups, col_groups = detect_panel_boundaries(arr)
    row_boxes = extract_panel_boxes(row_groups)
    col_boxes = extract_panel_boxes(col_groups)
    if not row_boxes or not col_boxes:
        raise RuntimeError(f"No panel boxes resolved from {grid_png}")

    x0, x1 = col_boxes[0]
    tiles: List[np.ndarray] = []
    step_indices = list(selected_step_indices) if selected_step_indices is not None else list(range(len(row_boxes)))
    for step_idx in step_indices:
        if step_idx < 0 or step_idx >= len(row_boxes):
            continue
        y0, y1 = row_boxes[step_idx]
        crop = arr[y0 : y1 + 1, x0 : x1 + 1]
        if border_px > 0:
            crop = np.pad(
                crop,
                ((border_px, border_px), (border_px, border_px), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        crop = _upscale_tile(crop, upscale)
        tiles.append(crop)
    return tiles, len(row_boxes)


def shorten_key_label(key: str, max_len: int = 38) -> str:
    value = str(key).strip()
    if "__" in value:
        value = value.rsplit("__", 1)[-1]
    value = value.replace("_", " ")
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def compose_multi_key_strip(
    rows: Sequence[Tuple[str, Sequence[np.ndarray]]],
    *,
    step_labels: Sequence[str],
    show_row_labels: bool,
    gap_px: int,
    row_gap_px: int,
) -> Tuple[np.ndarray, List[float], List[float]]:
    if not rows:
        raise RuntimeError("No attention-strip rows to compose")

    label_w = 280 if bool(show_row_labels) else 0
    top_label_h = 46 if step_labels and any(str(x).strip() for x in step_labels) else 0
    tile_h = max(tile.shape[0] for _, tiles in rows for tile in tiles)
    tile_w = max(tile.shape[1] for _, tiles in rows for tile in tiles)
    num_cols = max(len(tiles) for _, tiles in rows)
    canvas_h = top_label_h + len(rows) * tile_h + max(len(rows) - 1, 0) * max(int(row_gap_px), 0)
    canvas_w = label_w + num_cols * tile_w + max(num_cols - 1, 0) * max(int(gap_px), 0)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    x_centers: List[float] = []
    y_centers: List[float] = []

    for col_idx in range(num_cols):
        x = label_w + col_idx * (tile_w + max(int(gap_px), 0))
        x_centers.append(float(x + tile_w * 0.5))

    for row_idx, (_, tiles) in enumerate(rows):
        y = top_label_h + row_idx * (tile_h + max(int(row_gap_px), 0))
        y_centers.append(float(y + tile_h * 0.5))
        for col_idx, tile in enumerate(tiles):
            x = label_w + col_idx * (tile_w + max(int(gap_px), 0))
            yy = y + (tile_h - tile.shape[0]) // 2
            canvas[yy : yy + tile.shape[0], x : x + tile.shape[1]] = tile

    return canvas, x_centers, y_centers


def save_strip_figure(
    canvas: np.ndarray,
    step_labels: Sequence[str],
    x_centers: Sequence[float],
    row_labels: Sequence[str],
    y_centers: Sequence[float],
    title: str,
    out_path: Path,
    *,
    show_row_labels: bool,
    figure_dpi: int,
):
    ncols = len(step_labels) if step_labels else 1
    nrows = len(row_labels) if row_labels else 1
    fig_w = max(10.0, ncols * 1.65 + 2.8)
    fig_h = max(2.6 + nrows * 1.55, 3.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(canvas)
    ax.set_axis_off()
    if str(title).strip():
        ax.set_title(title, fontsize=13, pad=10)
    if step_labels and any(str(x).strip() for x in step_labels):
        label_y = 12
        for idx, label in enumerate(step_labels):
            if not str(label).strip():
                continue
            x = float(x_centers[idx]) if idx < len(x_centers) else float(canvas.shape[1] * (idx + 0.5) / max(ncols, 1))
            ax.text(
                x,
                label_y,
                f"step {label}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "white", "alpha": 0.96},
            )
    if bool(show_row_labels):
        for idx, label in enumerate(row_labels):
            if not str(label).strip():
                continue
            y = float(y_centers[idx]) if idx < len(y_centers) else float(canvas.shape[0] * (idx + 0.5) / max(nrows, 1))
            ax.text(
                266,
                y,
                label,
                ha="right",
                va="center",
                fontsize=10,
                color="black",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "white", "alpha": 0.96},
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=max(int(figure_dpi), 120), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_focus_step_values(samples, cohort_specs, group_name: str, metric_name: str) -> Tuple[int, ...]:
    step_grid, _ = most_common_grid(samples, group_name, metric_name)
    if step_grid is None:
        raise RuntimeError(f"Failed to resolve common step grid for group={group_name} metric={metric_name}")
    return tuple(int(x) for x in step_grid)


def collect_first_block_stats(
    samples,
    cohort_specs: Sequence[CohortSpec],
    *,
    group_name: str,
    metric_name: str,
    focus_block_idx: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    step_grid, block_grid = most_common_grid(samples, group_name, metric_name)
    if step_grid is None or block_grid is None:
        raise RuntimeError(f"Failed to resolve common grid for group={group_name} metric={metric_name}")
    if focus_block_idx < 0 or focus_block_idx >= len(block_grid):
        raise IndexError(
            f"focus_block_idx={focus_block_idx} is out of range for {len(block_grid)} blocks "
            f"(values={list(block_grid)})"
        )

    step_vals = np.asarray(step_grid, dtype=np.int64)
    stats: List[Dict[str, object]] = []
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
        line = stack[:, :, focus_block_idx]
        mean, sem = nanmean_sem(line)
        stats.append(
            {
                "cohort_name": cohort.name,
                "cohort_label": cohort.label,
                "cohort_color": cohort.color,
                "block_idx": int(focus_block_idx),
                "block_value": int(block_grid[focus_block_idx]),
                "n": int(stack.shape[0]),
                "mean": mean,
                "sem": sem,
            }
        )
    if not stats:
        raise RuntimeError(f"No cohort statistics resolved for group={group_name} metric={metric_name}")
    return step_vals, stats


def choose_step_ticks(step_vals: np.ndarray, target: int = 8) -> np.ndarray:
    if step_vals.size <= target:
        return step_vals
    idx = np.linspace(0, step_vals.size - 1, num=target)
    idx = np.unique(np.round(idx).astype(np.int64))
    return step_vals[idx]


def plot_focus_metric(
    step_vals: np.ndarray,
    stats: Sequence[Dict[str, object]],
    out_path: Path,
    *,
    metric_name: str,
    group_name: str,
    line_width: float,
    figure_dpi: int,
):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    for item in stats:
        mean = np.asarray(item["mean"], dtype=np.float64)
        sem = np.asarray(item["sem"], dtype=np.float64)
        label = str(item["cohort_label"])
        color = str(item["cohort_color"])
        ax.plot(step_vals, mean, color=color, linewidth=float(line_width), marker="o", markersize=3.6, label=label)
        ax.fill_between(step_vals, mean - sem, mean + sem, color=color, alpha=0.14)

    first_block_value = int(stats[0]["block_value"])
    ax.set_xlabel("Denoising step")
    ax.set_ylabel(f"{group_name} {metric_name}")
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.7)
    ax.set_xticks(choose_step_ticks(step_vals))
    ax.legend(frameon=True, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=max(int(figure_dpi), 120), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_attention_strips(
    *,
    cohort_specs: Sequence[CohortSpec],
    grid_roots: Sequence[Path],
    output_dir: Path,
    export_formats: Sequence[str],
    step_label_mode: str,
    keys_per_cohort: int,
    step_sample_count: int,
    strip_gap_px: int,
    strip_row_gap_px: int,
    strip_border_px: int,
    strip_upscale: int,
    figure_dpi: int,
    metric_step_values: Optional[Sequence[int]],
) -> List[Dict[str, object]]:
    strip_dir = output_dir / "attention_strips"
    records: List[Dict[str, object]] = []
    for cohort in cohort_specs:
        selected_pairs, tried = select_keys_for_cohort(
            grid_roots,
            cohort,
            max_keys=int(keys_per_cohort),
        )
        record: Dict[str, object] = {
            "cohort_name": cohort.name,
            "cohort_label": cohort.label,
            "key_file": str(cohort.key_file),
            "tried_keys": tried,
        }
        if not selected_pairs:
            record["status"] = "missing_grid_png"
            records.append(record)
            continue

        first_tiles, row_count = extract_first_block_tiles(
            selected_pairs[0][1],
            border_px=int(strip_border_px),
            upscale=int(strip_upscale),
            selected_step_indices=None,
        )
        del first_tiles

        sampled_step_indices = sample_indices_evenly(row_count, int(step_sample_count))
        label_values = None
        if metric_step_values is not None and len(metric_step_values) > 0:
            if len(metric_step_values) == row_count:
                label_values = [int(metric_step_values[idx]) for idx in sampled_step_indices]
        step_labels = step_labels_for_strip(
            len(sampled_step_indices),
            mode=step_label_mode,
            step_values=label_values,
        )

        rows: List[Tuple[str, Sequence[np.ndarray]]] = []
        selected_items: List[Dict[str, object]] = []
        for row_idx, (key, grid_png) in enumerate(selected_pairs, start=1):
            tiles, cur_row_count = extract_first_block_tiles(
                grid_png,
                border_px=int(strip_border_px),
                upscale=int(strip_upscale),
                selected_step_indices=sampled_step_indices,
            )
            if cur_row_count != row_count:
                raise RuntimeError(
                    f"Inconsistent step count inside cohort={cohort.name}: base={row_count}, key={key}, got={cur_row_count}"
                )
            rows.append((f"{row_idx}. {shorten_key_label(key)}", tiles))
            selected_items.append(
                {
                    "key": key,
                    "grid_png": str(grid_png),
                    "resolved_grid_root": str(grid_png.parent.parent),
                }
            )

        canvas, x_centers, y_centers = compose_multi_key_strip(
            rows,
            step_labels=step_labels,
            show_row_labels=False,
            gap_px=int(strip_gap_px),
            row_gap_px=int(strip_row_gap_px),
        )

        for fmt in export_formats:
            out_path = strip_dir / cohort.name / f"{cohort.name}_first_block_multi_key_sampled_steps.{fmt}"
            save_strip_figure(
                canvas,
                step_labels,
                x_centers,
                [label for label, _ in rows],
                y_centers,
                "",
                out_path,
                show_row_labels=False,
                figure_dpi=int(figure_dpi),
            )

        record.update(
            {
                "status": "ok",
                "selected_keys": selected_items,
                "num_selected_keys": len(selected_items),
                "num_detected_steps": row_count,
                "sampled_step_indices": sampled_step_indices,
                "num_sampled_steps": len(sampled_step_indices),
                "label_mode": step_label_mode,
            }
        )
        records.append(record)
    return records


def main():
    args = parse_args()
    selected_long_csv = Path(args.selected_long_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_formats = normalize_formats(args.export_format)
    grid_roots = resolve_grid_roots(args)
    cohort_specs = build_qwen_cohort_specs(args)

    samples = load_samples_from_long_csv(
        long_csv=selected_long_csv,
        cohort_specs=cohort_specs,
        group_names=(str(args.metric_group),),
        metric_names=(str(args.metric_name),),
    )
    if not samples:
        raise RuntimeError(f"No samples loaded from {selected_long_csv}")

    step_vals, stats = collect_first_block_stats(
        samples,
        cohort_specs,
        group_name=str(args.metric_group),
        metric_name=str(args.metric_name),
        focus_block_idx=int(args.focus_block_idx),
    )

    metric_dir = output_dir / "metric_focus"
    metric_base = f"{sanitize_filename(str(args.metric_group))}_{sanitize_filename(str(args.metric_name))}_block{int(args.focus_block_idx)}"
    for fmt in export_formats:
        plot_focus_metric(
            step_vals,
            stats,
            metric_dir / f"{metric_base}.{fmt}",
            metric_name=str(args.metric_name),
            group_name=str(args.metric_group),
            line_width=float(args.line_width),
            figure_dpi=int(args.figure_dpi),
        )

    attention_records = export_attention_strips(
        cohort_specs=cohort_specs,
        grid_roots=grid_roots,
        output_dir=output_dir,
        export_formats=export_formats,
        step_label_mode=str(args.step_label_mode),
        keys_per_cohort=int(args.keys_per_cohort),
        step_sample_count=int(args.step_sample_count),
        strip_gap_px=int(args.strip_gap_px),
        strip_row_gap_px=int(args.strip_row_gap_px),
        strip_border_px=int(args.strip_border_px),
        strip_upscale=int(args.strip_upscale),
        figure_dpi=int(args.figure_dpi),
        metric_step_values=step_vals,
    )

    meta = {
        "selected_long_csv": str(selected_long_csv),
        "output_dir": str(output_dir),
        "grid_roots": [str(x) for x in grid_roots],
        "export_formats": list(export_formats),
        "metric_group": str(args.metric_group),
        "metric_name": str(args.metric_name),
        "focus_block_idx": int(args.focus_block_idx),
        "keys_per_cohort": int(args.keys_per_cohort),
        "step_sample_count": int(args.step_sample_count),
        "step_label_mode": str(args.step_label_mode),
        "strip_upscale": int(args.strip_upscale),
        "figure_dpi": int(args.figure_dpi),
        "metric_step_values": [int(x) for x in step_vals.tolist()],
        "cohorts": [
            {
                "name": item.name,
                "label": item.label,
                "key_file": str(item.key_file),
                "color": item.color,
            }
            for item in cohort_specs
        ],
        "attention_records": attention_records,
        "metric_stats": [
            {
                "cohort_name": str(item["cohort_name"]),
                "cohort_label": str(item["cohort_label"]),
                "block_idx": int(item["block_idx"]),
                "block_value": int(item["block_value"]),
                "n": int(item["n"]),
            }
            for item in stats
        ],
    }
    (output_dir / "paper_figure_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"[done] output_dir={output_dir}")
    log(f"[done] metric_focus={metric_dir / metric_base}")
    log(f"[done] attention_strips={output_dir / 'attention_strips'}")


if __name__ == "__main__":
    main()

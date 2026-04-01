#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare attention-map metrics across an arbitrary number of key cohorts.

By default, every `*.txt` file under `--cohort-dir` becomes one cohort:
- the cohort name/label is the txt stem (filename without `.txt`)
- all keys in that file are included
- pairwise difference heatmaps are saved separately for every cohort pair

Example:
python /data/benchmark_metrics/insight/attention_metrics_compare_key_groups_multi.py \
    --root-dir /mnt/jfs/qwen-edit-attn-fullmap-keycolor-save-kfull-1-1 \
    --output-dir /data/benchmark_metrics/logs/qwen_attn_key_group_compare_multi-1-1 \
    --cohort-dir /data/benchmark_metrics/insight/key_folder \
    --summary-csv /data/benchmark_metrics/logs/attention_metrics_summary.csv



"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_hex

try:
    from insight.attention_metrics_compare_key_groups import (
        CohortSpec,
        DEFAULT_GROUP_NAMES,
        DEFAULT_METRIC_NAMES,
        collect_samples,
        export_summary_csv,
        load_samples_from_long_csv,
        log,
        most_common_grid,
        plot_block_profiles,
        plot_cohort_heatmaps,
        plot_first_last_block,
        sanitize_filename,
        stack_by_cohort,
        write_long_and_agg_csvs,
        write_sample_manifest,
    )
except ModuleNotFoundError:
    from attention_metrics_compare_key_groups import (
        CohortSpec,
        DEFAULT_GROUP_NAMES,
        DEFAULT_METRIC_NAMES,
        collect_samples,
        export_summary_csv,
        load_samples_from_long_csv,
        log,
        most_common_grid,
        plot_block_profiles,
        plot_cohort_heatmaps,
        plot_first_last_block,
        sanitize_filename,
        stack_by_cohort,
        write_long_and_agg_csvs,
        write_sample_manifest,
    )


DEFAULT_COHORT_DIR = Path("/data/benchmark_metrics/insight/key_folder")
PALETTE_NAMES = ("tab20", "tab20b", "tab20c")


def parse_args():
    parser = argparse.ArgumentParser("Compare attention metrics across an arbitrary number of key cohorts")
    parser.add_argument(
        "--root-dir",
        default="/mnt/jfs/qwen-edit-attn-fullmap-keycolor-save",
        help="Root directory that contains *_attn folders",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/benchmark_metrics/insight/output/qwen_attn_key_group_compare_multi",
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
    if not cohort_dir.is_dir():
        raise FileNotFoundError(f"cohort-dir does not exist: {cohort_dir}")
    txt_files = sorted(path for path in cohort_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
    if not txt_files:
        raise RuntimeError(f"No *.txt cohort files found under {cohort_dir}")
    return txt_files


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


def plot_pairwise_difference_heatmaps(
    metric_name: str,
    group_names: Sequence[str],
    cohort_specs: Sequence[CohortSpec],
    samples,
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

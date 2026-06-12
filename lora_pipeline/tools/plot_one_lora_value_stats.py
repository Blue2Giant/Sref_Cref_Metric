#!/usr/bin/env python3
"""
Plot per-key value-count statistics for one-lora JSONL files.

Expected inputs inside the source directory:
  <base_model>_content_one_lora.jsonl
  <base_model>_style_one_lora.jsonl

Each JSONL row should look like:
  {"<key>": ["value1", "value2", ...]}

Outputs:
  - <output_prefix>.png
  - <output_prefix>.svg
  - <output_prefix>_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


DEFAULT_INPUT_DIR = Path("/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plots"
DEFAULT_OUTPUT_PREFIX = "one_lora_value_stats"

KIND_ORDER = ("content", "style")
KIND_LABEL = {
    "content": "Content",
    "style": "Style",
}
PALETTE = {
    "content": "#2A9D8F",
    "style": "#E76F51",
}
BACKGROUND = "#F8F5F0"
PANEL_FACE = "#FFFDFC"
GRID_COLOR = "#D8D4CC"
TEXT_COLOR = "#2B2B2B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot average values-per-key and per-key distributions for one-lora JSONLs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory that contains *_content_one_lora.jsonl and *_style_one_lora.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for figures and summary CSV",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output file prefix without extension",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="PNG export DPI",
    )
    return parser.parse_args()


def discover_input_files(input_dir: Path) -> Dict[str, Dict[str, Path]]:
    pattern = re.compile(r"^(?P<base>.+)_(?P<kind>content|style)_one_lora\.jsonl$")
    discovered: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for path in sorted(input_dir.glob("*_one_lora.jsonl")):
        match = pattern.match(path.name)
        if not match:
            continue
        base = match.group("base")
        kind = match.group("kind")
        discovered[base][kind] = path
    return dict(discovered)


def normalize_base_label(base_model: str) -> str:
    if base_model.lower() == "flux":
        return "FLUX"
    if base_model.lower() == "qwen":
        return "Qwen"
    return base_model.replace("_", " ").title()


def iter_valid_rows(path: Path) -> Iterable[Tuple[str, Sequence[object]]]:
    with path.open("r", encoding="utf-8") as fin:
        for line_number, raw_line in enumerate(fin, start=1):
            line = (raw_line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(obj, dict) or len(obj) != 1:
                raise ValueError(f"expected a single-entry object at {path}:{line_number}")
            key, values = next(iter(obj.items()))
            if not isinstance(key, str) or not key:
                raise ValueError(f"expected non-empty key at {path}:{line_number}")
            if not isinstance(values, list):
                raise ValueError(f"expected list values at {path}:{line_number}")
            yield key, values


def build_dataframes(input_files: Dict[str, Dict[str, Path]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_records: List[Dict[str, object]] = []
    summary_records: List[Dict[str, object]] = []

    for base_model in sorted(input_files):
        kind_map = input_files[base_model]
        for kind in KIND_ORDER:
            if kind not in kind_map:
                raise ValueError(f"missing {kind!r} JSONL for base model {base_model!r}")
            path = kind_map[kind]
            counts: List[int] = []
            for key, values in iter_valid_rows(path):
                value_count = len(values)
                counts.append(value_count)
                long_records.append(
                    {
                        "base_model": base_model,
                        "base_label": normalize_base_label(base_model),
                        "kind": kind,
                        "kind_label": KIND_LABEL[kind],
                        "key": key,
                        "value_count": value_count,
                    }
                )
            if not counts:
                raise ValueError(f"no valid rows found in {path}")
            summary_records.append(
                {
                    "base_model": base_model,
                    "base_label": normalize_base_label(base_model),
                    "kind": kind,
                    "kind_label": KIND_LABEL[kind],
                    "key_count": len(counts),
                    "avg_value_count": mean(counts),
                    "median_value_count": median(counts),
                    "min_value_count": min(counts),
                    "max_value_count": max(counts),
                    "total_value_count": sum(counts),
                }
            )

    long_df = pd.DataFrame(long_records)
    summary_df = pd.DataFrame(summary_records)
    return long_df, summary_df


def build_tick_labels(summary_df: pd.DataFrame, base_order: Sequence[str]) -> List[str]:
    del summary_df
    return [normalize_base_label(base_model) for base_model in base_order]


def plot_figure(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
    dpi: int,
):
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.facecolor": PANEL_FACE,
            "figure.facecolor": BACKGROUND,
            "axes.edgecolor": "#C5C0B6",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "text.color": TEXT_COLOR,
        },
    )

    base_order = list(sorted(summary_df["base_model"].unique()))
    tick_labels = build_tick_labels(summary_df, base_order)

    fig = plt.figure(figsize=(17.2, 7.4), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.24)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_total = fig.add_subplot(grid[0, 1])

    x = np.arange(len(base_order))
    width = 0.34
    max_mean = float(summary_df["avg_value_count"].max())

    for offset, kind in [(-width / 2, "content"), (width / 2, "style")]:
        subset = (
            summary_df[summary_df["kind"] == kind]
            .set_index("base_model")
            .loc[base_order]
            .reset_index()
        )
        bars = ax_bar.bar(
            x + offset,
            subset["avg_value_count"].to_numpy(),
            width=width,
            color=PALETTE[kind],
            edgecolor="#FFFFFF",
            linewidth=1.8,
            alpha=0.95,
        )
        for bar, row in zip(bars, subset.itertuples()):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_mean * 0.03,
                f"{row.avg_value_count:.1f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=TEXT_COLOR,
            )

    sns.violinplot(
        data=long_df,
        x="base_model",
        y="value_count",
        hue="kind",
        order=base_order,
        hue_order=list(KIND_ORDER),
        palette=PALETTE,
        split=True,
        inner="quart",
        cut=0,
        bw_adjust=0.9,
        linewidth=1.25,
        saturation=1.0,
        density_norm="width",
        ax=ax_total,
    )

    legend_handles = [
        Patch(facecolor=PALETTE["content"], edgecolor="none", label="Content"),
        Patch(facecolor=PALETTE["style"], edgecolor="none", label="Style"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        fontsize=14,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tick_labels, fontsize=15, fontweight="semibold")
    ax_bar.set_ylabel("Average Reference", labelpad=14, fontsize=17, fontweight="semibold")
    ax_bar.set_xlabel("")
    ax_bar.set_ylim(0, max_mean * 1.28)
    ax_bar.grid(axis="y", alpha=0.55)
    ax_bar.grid(axis="x", visible=False)
    ax_bar.tick_params(axis="y", labelsize=14)

    ax_total.set_xticks(x)
    ax_total.set_xticklabels(tick_labels, fontsize=15, fontweight="semibold")
    ax_total.set_xlabel("")
    ax_total.set_ylabel("Total Reference", labelpad=14, fontsize=17, fontweight="semibold")
    ax_total.grid(axis="y", alpha=0.55)
    ax_total.grid(axis="x", visible=False)
    ax_total.tick_params(axis="y", labelsize=14)
    if ax_total.legend_ is not None:
        ax_total.legend_.remove()

    for ax in (ax_bar, ax_total):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.55)
        ax.spines["bottom"].set_alpha(0.55)

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.83, wspace=0.24)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary_csv(summary_df: pd.DataFrame, path: Path):
    fieldnames = [
        "base_model",
        "kind",
        "key_count",
        "avg_value_count",
        "median_value_count",
        "min_value_count",
        "max_value_count",
        "total_value_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_df.sort_values(["base_model", "kind"]).to_dict("records"):
            writer.writerow({name: row[name] for name in fieldnames})


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(input_dir)
    if not input_files:
        raise SystemExit(f"no matching one-lora JSONLs found in {input_dir}")

    long_df, summary_df = build_dataframes(input_files)

    output_png = out_dir / f"{args.output_prefix}.png"
    output_svg = out_dir / f"{args.output_prefix}.svg"
    summary_csv = out_dir / f"{args.output_prefix}_summary.csv"

    plot_figure(
        long_df=long_df,
        summary_df=summary_df,
        output_png=output_png,
        output_svg=output_svg,
        dpi=args.dpi,
    )
    write_summary_csv(summary_df=summary_df, path=summary_csv)

    print(f"png={output_png}")
    print(f"svg={output_svg}")
    print(f"summary_csv={summary_csv}")
    print()
    print(summary_df.sort_values(['base_model', 'kind']).to_string(index=False))


if __name__ == "__main__":
    main()

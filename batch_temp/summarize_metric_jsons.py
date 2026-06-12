#!/usr/bin/env python3
"""Summarize metric scores across multiple model output dirs.

For each MODEL passed on the command line, looks up the standard set of metric
JSONs under <root>/<MODEL>/ (dinov2, cas, oneig, clipcap, csd, laion, v25, VLM
style/content/follow, qwen triplet judges), computes the mean score per metric,
and writes a single combined CSV/markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

METRIC_FILES: List[Tuple[str, str]] = [
    ("dinov2", "dinov2_out.json"),
    ("cas", "cas_out.json"),
    ("oneig", "oneig_out.json"),
    ("clipcap", "clipcap_out.json"),
    ("csd", "csd_out.json"),
    ("laion_aes", "laion_scores.json"),
    ("v25_aes", "v25_scores.json"),
    ("vlm_style", "qwen_resize_output_style_descrete.json"),
    ("vlm_content", "qwen_resize_output_content_descrete.json"),
    ("vlm_follow", "follow_scores.json"),
    ("qwen_reject_cref", "qwen_reject_cref.json"),
    ("qwen_reject_sref", "qwen_reject_sref.json"),
]


def _iter_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_numbers(item)
        return
    if isinstance(value, dict):
        if "score" in value and isinstance(value["score"], (int, float)):
            yield float(value["score"])
            return
        for v in value.values():
            yield from _iter_numbers(v)


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def _compute_metric_mean(path: str) -> Tuple[Optional[float], int]:
    """Return (mean, n_values). mean is None if file missing."""
    if not os.path.exists(path):
        return None, 0
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return None, 0
    values: List[float] = []
    if isinstance(data, dict):
        for v in data.values():
            values.extend(list(_iter_numbers(v)))
    elif isinstance(data, list):
        for v in data:
            values.extend(list(_iter_numbers(v)))
    if not values:
        return None, 0
    return _mean(values), len(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="parent dir containing each MODEL subdir")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    # rows[model] = {metric: (mean, n)}
    rows: Dict[str, Dict[str, Tuple[Optional[float], int]]] = {}
    for model in args.models:
        per_metric: Dict[str, Tuple[Optional[float], int]] = {}
        for metric_name, filename in METRIC_FILES:
            path = os.path.join(args.root, model, filename)
            mean, n = _compute_metric_mean(path)
            per_metric[metric_name] = (mean, n)
        rows[model] = per_metric

    metric_names = [m for m, _ in METRIC_FILES]

    # plain text table
    def fmt(val: Optional[float]) -> str:
        if val is None:
            return "---"
        return f"{val:.4f}"

    col_w = max(8, max(len(m) for m in metric_names) + 2)
    name_w = max(8, max(len(m) for m in args.models) + 2)
    header = "model".ljust(name_w) + "".join(m.ljust(col_w) for m in metric_names)
    print(header)
    print("-" * len(header))
    for model, per_metric in rows.items():
        line = model.ljust(name_w)
        for m in metric_names:
            mean, _n = per_metric[m]
            line += fmt(mean).ljust(col_w)
        print(line)

    # n table
    print()
    print("n (number of items aggregated):")
    print("model".ljust(name_w) + "".join(m.ljust(col_w) for m in metric_names))
    for model, per_metric in rows.items():
        line = model.ljust(name_w)
        for m in metric_names:
            _mean, n = per_metric[m]
            line += str(n).ljust(col_w)
        print(line)

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model"] + metric_names)
            for model, per_metric in rows.items():
                w.writerow([model] + [
                    (fmt(per_metric[m][0]) if per_metric[m][0] is not None else "")
                    for m in metric_names
                ])
        print(f"\nwrote {args.out_csv}")

    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write("| model | " + " | ".join(metric_names) + " |\n")
            f.write("|" + "|".join(["---"] * (1 + len(metric_names))) + "|\n")
            for model, per_metric in rows.items():
                f.write("| " + model + " | " + " | ".join(
                    fmt(per_metric[m][0]) for m in metric_names
                ) + " |\n")
        print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()

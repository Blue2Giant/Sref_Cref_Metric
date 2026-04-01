#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/benchmark_metrics/insight/attention_metrics_export_csv.py \
    --root-dir  /mnt/jfs/qwen-edit-attn-fullmap-keycolor-save-kfull-1-1  \
    --out-csv /data/benchmark_metrics/logs/attention_metrics_summary.csv \
    --device cuda \
    --groups text,cref,sref \
    --topk 5 \
    --reduction mean

python /data/benchmark_metrics/insight/attention_metrics_export_csv.py \
    --root-dir  /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/flux-klein-9b-attn-fullmap-1-1  \
    --out-csv /data/benchmark_metrics/logs/flux_attention_metrics_summary-1-1.csv \
    --device cuda \
    --groups text,cref,sref \
    --topk 5 \
    --reduction mean
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

try:
    from insight.attention_metrics import (
        DEFAULT_GROUP_NAMES,
        compute_metrics_for_named_groups_from_pt,
        get_k_slice_from_meta,
        summarize_group_metrics,
    )
except ModuleNotFoundError:
    from attention_metrics import (
        DEFAULT_GROUP_NAMES,
        compute_metrics_for_named_groups_from_pt,
        get_k_slice_from_meta,
        summarize_group_metrics,
    )


def parse_args():
    parser = argparse.ArgumentParser("Export attention metrics from attention_step_block_grid.pt files to CSV")
    parser.add_argument(
        "--root-dir",
        default="/mnt/jfs/qwen-edit-attn-fullmap-keycolor-save",
        help="Root directory that contains *_attn folders",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path. Default: <root-dir>/attention_metrics_summary.csv",
    )
    parser.add_argument("--device", default="auto", help='Device such as "auto", "cpu", "cuda", or "cuda:0"')
    parser.add_argument("--dtype", default="float32", help='Compute dtype such as "float32", "float16", or "bfloat16"')
    parser.add_argument("--groups", default="text,cref,sref", help="Comma-separated named groups")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--reduction", choices=["mean", "median", "max", "min"], default="mean")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N pt files for debugging")
    parser.add_argument("--verbose", action="store_true", help="Print one progress line per file")
    return parser.parse_args()


def iter_pt_files(root_dir: Path) -> List[Path]:
    pt_files = sorted(root_dir.rglob("attention_step_block_grid.pt"))
    return [path for path in pt_files if path.is_file()]


def _safe_int(value, default=-1) -> int:
    if value is None:
        return int(default)
    return int(value)


def _safe_name_from_parent(parent_name: str) -> str:
    if parent_name.endswith("_attn"):
        return parent_name[: -len("_attn")]
    return parent_name


def _add_group_range_columns(
    row: Dict[str, object],
    group_name: str,
    meta: Dict[str, object],
):
    row[f"{group_name}_full_k_start"] = _safe_int(meta.get("full_start"))
    row[f"{group_name}_full_k_end_exclusive"] = _safe_int(meta.get("full_end_exclusive"))
    row[f"{group_name}_full_k_end_inclusive"] = _safe_int(meta.get("full_end_inclusive"))
    row[f"{group_name}_full_k_length"] = _safe_int(meta.get("full_length"), default=0)
    row[f"{group_name}_sampled_k_start"] = _safe_int(meta.get("sample_start"))
    row[f"{group_name}_sampled_k_end_exclusive"] = _safe_int(meta.get("sample_end_exclusive"))
    row[f"{group_name}_sampled_k_end_inclusive"] = _safe_int(meta.get("sample_end_inclusive"))
    row[f"{group_name}_sampled_k_length"] = _safe_int(meta.get("sample_length"), default=0)


def _add_group_metric_columns(
    row: Dict[str, object],
    group_name: str,
    summary: Dict[str, float],
):
    for metric_name, value in summary.items():
        row[f"{group_name}_{metric_name}"] = float(value)


def build_row(
    pt_path: Path,
    group_names: Sequence[str],
    reduction: str,
    device: str,
    dtype: str,
    topk: int,
) -> Dict[str, object]:
    batch, metrics = compute_metrics_for_named_groups_from_pt(
        pt_path=pt_path,
        device=device,
        dtype=dtype,
        group_names=group_names,
        use_sample=True,
        topk=topk,
    )
    summary = summarize_group_metrics(metrics, reduction=reduction)
    payload = batch.payload
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


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"root-dir does not exist: {root_dir}")
    out_csv = Path(args.out_csv) if args.out_csv else root_dir / "attention_metrics_summary.csv"
    group_names = tuple(x.strip() for x in str(args.groups).split(",") if x.strip()) or DEFAULT_GROUP_NAMES

    pt_files = iter_pt_files(root_dir)
    if int(args.limit) > 0:
        pt_files = pt_files[: int(args.limit)]
    if not pt_files:
        raise RuntimeError(f"No attention_step_block_grid.pt files found under {root_dir}")

    print(f"root_dir={root_dir}")
    print(f"out_csv={out_csv}")
    print(f"num_pt_files={len(pt_files)}")
    print(f"device={args.device} dtype={args.dtype} groups={group_names} reduction={args.reduction}")

    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for idx, pt_path in enumerate(pt_files, start=1):
        try:
            row = build_row(
                pt_path=pt_path,
                group_names=group_names,
                reduction=str(args.reduction),
                device=str(args.device),
                dtype=str(args.dtype),
                topk=int(args.topk),
            )
            rows.append(row)
            if args.verbose:
                print(f"[{idx}/{len(pt_files)}] ok {pt_path.parent.name}")
        except Exception as exc:
            failures.append(f"{pt_path}\t{type(exc).__name__}\t{exc}")
            print(f"[{idx}/{len(pt_files)}] fail {pt_path} :: {type(exc).__name__}: {exc}")
        finally:
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("All files failed; no CSV written")

    fieldnames = collect_fieldnames(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows_written={len(rows)}")
    print(f"csv_saved={out_csv}")
    if failures:
        fail_path = out_csv.with_suffix(out_csv.suffix + ".failures.txt")
        fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"failures={len(failures)}")
        print(f"failure_log={fail_path}")
    else:
        print("failures=0")


if __name__ == "__main__":
    main()

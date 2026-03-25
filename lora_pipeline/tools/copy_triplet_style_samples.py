#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/benchmark_metrics/lora_pipeline/tools/copy_triplet_style_samples.py \
  --output-root /mnt/jfs/lora_combine/triplet_style_copy_debug \
  --sample-count 100 \
  --result-label 1 \
  --jpg-quality 78 \
  --overwrite
python /data/benchmark_metrics/lora_pipeline/tools/copy_triplet_style_samples.py \
  --binary-jsonl /data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325/style_firsthit.jsonl \
  --output-root /mnt/jfs/lora_combine/triplet_style_copy_debug_firsthit \
  --sample-count 300 \
  --result-label 1 \
  --jpg-quality 78 \
  --overwrite
python /data/benchmark_metrics/lora_pipeline/tools/copy_triplet_style_samples.py \
  --binary-jsonl /data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.4_2/style_firsthit.jsonl \
  --output-root /mnt/jfs/lora_combine/triplet_style_copy_debug_firsthit_0325_0.4_2 \
  --sample-count 300 \
  --result-label 1 \
  --jpg-quality 78 \
  --overwrite
"""
import argparse
import concurrent.futures
import io
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image
from megfile import smart_exists, smart_open


@dataclass
class SampleRecord:
    pair_key: str
    label: int
    triplet_paths: List[str]
    style_paths: List[str]


def log(msg: str):
    print(msg, flush=True)


def read_binary_labels(path: str) -> Dict[str, Dict[str, Optional[str]]]:
    out: Dict[str, Dict[str, Optional[str]]] = {}
    if not os.path.isfile(path):
        raise RuntimeError(f"判别结果文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                if not isinstance(k, str):
                    continue
                if v in (0, 1):
                    out[k] = {"label": int(v), "firsthit_path": None}
                elif isinstance(v, str):
                    vv = v.strip()
                    out[k] = {"label": 1 if vv else 0, "firsthit_path": vv or None}
    return out


def read_triplet_paths(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not os.path.isfile(path):
        raise RuntimeError(f"triplet文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                if not isinstance(k, str) or not isinstance(v, list):
                    continue
                paths = [str(x).strip() for x in v if isinstance(x, str) and str(x).strip()]
                if paths:
                    out[k] = paths
    return out


def read_style_paths(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not os.path.isfile(path):
        raise RuntimeError(f"style索引文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                if not isinstance(k, str) or not isinstance(v, list):
                    continue
                paths = [str(x).strip() for x in v if isinstance(x, str) and str(x).strip()]
                if paths:
                    out[k] = paths
    return out


def read_image_bytes(path: str) -> bytes:
    with smart_open(path, "rb") as f:
        return f.read()


def save_jpg(src_path: str, dst_path: str, jpg_quality: int):
    raw = read_image_bytes(src_path)
    with Image.open(io.BytesIO(raw)) as img:
        rgb = img.convert("RGB")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        rgb.save(dst_path, format="JPEG", quality=int(jpg_quality), optimize=True)


def _copy_task(src: str, dst: str, jpg_quality: int, pair_key: str) -> Tuple[bool, Dict[str, str]]:
    try:
        if not smart_exists(src):
            raise RuntimeError(f"source_not_found: {src}")
        save_jpg(src, dst, jpg_quality)
        return True, {}
    except Exception as e:
        return False, {"pair_key": pair_key, "src": src, "dst": dst, "err": str(e)}


def build_samples(
    labels: Dict[str, Dict[str, Optional[str]]],
    triplets: Dict[str, List[str]],
    styles: Dict[str, List[str]],
    target_label: int,
) -> Tuple[List[SampleRecord], Dict[str, int]]:
    records: List[SampleRecord] = []
    stats = {
        "total_labels": len(labels),
        "skip_no_triplet": 0,
        "skip_bad_key": 0,
        "skip_no_style_index": 0,
        "matched": 0,
    }
    for pair_key, payload in labels.items():
        label = int(payload.get("label", -1))
        if label != int(target_label):
            continue
        triplet_paths = triplets.get(pair_key)
        if not triplet_paths:
            stats["skip_no_triplet"] += 1
            continue
        firsthit_path = (payload.get("firsthit_path") or "").strip()
        if firsthit_path:
            style_paths = [firsthit_path]
            records.append(
                SampleRecord(
                    pair_key=pair_key,
                    label=label,
                    triplet_paths=triplet_paths,
                    style_paths=style_paths,
                )
            )
            stats["matched"] += 1
            continue
        if label == 0:
            records.append(
                SampleRecord(
                    pair_key=pair_key,
                    label=label,
                    triplet_paths=triplet_paths,
                    style_paths=[],
                )
            )
            stats["matched"] += 1
            continue
        if "__" not in pair_key:
            stats["skip_bad_key"] += 1
            continue
        _content_id, style_id = pair_key.split("__", 1)
        style_id = style_id.strip()
        style_paths = styles.get(style_id)
        if not style_paths:
            stats["skip_no_style_index"] += 1
            continue
        records.append(
            SampleRecord(
                pair_key=pair_key,
                label=label,
                triplet_paths=triplet_paths,
                style_paths=style_paths,
            )
        )
        stats["matched"] += 1
    return records, stats


def main():
    parser = argparse.ArgumentParser(description="按风格判别结果抽样并拷贝triplet/style图片到观察目录（转JPG）")
    parser.add_argument("--binary-jsonl", default="/data/benchmark_metrics/logs/triplet_style_index_judge_0324/style_binary.jsonl")
    parser.add_argument("--triplet-jsonl", default="/data/benchmark_metrics/logs/triplets_style_and_content_only.jsonl")
    parser.add_argument("--style-index-jsonl", default="/data/benchmark_metrics/logs/selections_with_origin_style_flux0325.jsonl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-count", type=int, default=200, help="随机抽样数量，0表示全量")
    parser.add_argument("--result-label", type=int, choices=[0, 1], required=True, help="筛选判别结果（0或1）")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=82,
        help="JPG质量(1-95，推荐 70-90；越高体积越大画质越好)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=32, help="并发拷贝线程数，默认32")
    args = parser.parse_args()

    if not (1 <= int(args.jpg_quality) <= 95):
        raise RuntimeError("--jpg-quality 必须在 1~95")

    labels = read_binary_labels(args.binary_jsonl)
    triplets = read_triplet_paths(args.triplet_jsonl)
    styles = read_style_paths(args.style_index_jsonl)
    samples, stats = build_samples(labels, triplets, styles, args.result_label)
    if not samples:
        raise RuntimeError("筛选后没有可处理样本")

    rng = random.Random(args.seed)
    if args.sample_count > 0 and args.sample_count < len(samples):
        samples = rng.sample(samples, args.sample_count)

    os.makedirs(args.output_root, exist_ok=True)

    copied = 0
    skipped = 0
    created_dir_set = set()
    fail_logs: List[Dict[str, str]] = []
    tasks: List[Tuple[str, str, str]] = []

    for rec in samples:
        for i, src in enumerate(rec.triplet_paths, start=1):
            sub = os.path.join(args.output_root, f"triplet_{i}")
            os.makedirs(sub, exist_ok=True)
            created_dir_set.add(sub)
            dst = os.path.join(sub, f"{rec.pair_key}.jpg")
            if (not args.overwrite) and os.path.isfile(dst):
                continue
            tasks.append((src, dst, rec.pair_key))

        for i, src in enumerate(rec.style_paths, start=1):
            sub = os.path.join(args.output_root, f"style_{i}")
            os.makedirs(sub, exist_ok=True)
            created_dir_set.add(sub)
            dst = os.path.join(sub, f"{rec.pair_key}.jpg")
            if (not args.overwrite) and os.path.isfile(dst):
                continue
            tasks.append((src, dst, rec.pair_key))

    workers = max(1, int(args.workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_copy_task, src, dst, int(args.jpg_quality), pair_key) for src, dst, pair_key in tasks]
        for fut in concurrent.futures.as_completed(futures):
            ok, err = fut.result()
            if ok:
                copied += 1
            else:
                skipped += 1
                fail_logs.append(err)

    report = {
        "binary_jsonl": args.binary_jsonl,
        "triplet_jsonl": args.triplet_jsonl,
        "style_index_jsonl": args.style_index_jsonl,
        "output_root": args.output_root,
        "sample_count": args.sample_count,
        "actual_samples": len(samples),
        "result_label": args.result_label,
        "seed": args.seed,
        "jpg_quality": args.jpg_quality,
        "stats": stats,
        "workers": workers,
        "scheduled_files": len(tasks),
        "copied_files": copied,
        "created_dirs": len(created_dir_set),
        "skipped_files": skipped,
    }
    report_path = os.path.join(args.output_root, "copy_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if fail_logs:
        err_path = os.path.join(args.output_root, "copy_errors.jsonl")
        with open(err_path, "w", encoding="utf-8") as f:
            for x in fail_logs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        log(f"[WARN] 有失败样本，错误日志: {err_path}")

    log(f"[DONE] sample={len(samples)} copied={copied} skipped={skipped} created_dirs={len(created_dir_set)}")
    log(f"[DONE] report={report_path}")


if __name__ == "__main__":
    main()

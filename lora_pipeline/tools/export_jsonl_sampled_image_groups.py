#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 JSONL 中的图片路径按随机采样导出到新的目录结构中，方便肉眼查看。

期望的常见输入形状:
1. 每行一个 dict，只有一个 key-value 对:
   {"<key_image_path>": ["<value1_image_path>", "<value2_image_path>"]}
2. value 也可以是单个字符串:
   {"<key_image_path>": "<value1_image_path>"}

输出目录示例:
  output_root/
    key/
      000001__00000_0__5d3c6a0f8b.jpg
    value1/
      000001__00000_0__5d3c6a0f8b.jpg
    value2/
      000001__00000_0__5d3c6a0f8b.jpg
    manifest.jsonl

同一条样本在 key/value1/value2/... 下会使用完全相同的 basename。
basename 会带采样序号和哈希，保证稳定且唯一。

示例:
python3 /data/benchmark_metrics/lora_pipeline/tools/export_jsonl_sampled_image_groups.py \
  --jsonl /mnt/jfs/logs/triplet_style_firsthit_judge_0325_0.5_2_2match_0328_per_image/style_firsthit_matched.jsonl \
  --output-root /mnt/jfs/logs/flux_style_firsthit_matched_2see_sample_200 \
  --sample-count 200 \
  --seed 42 \
  --jpg-quality 75 \
  --workers 32
python3 /data/benchmark_metrics/lora_pipeline/tools/export_jsonl_sampled_image_groups.py \
  --jsonl /mnt/jfs/logs/illustrious_triplet_style_firsthit_judge_0325_0.5_2_2match_0328_per_image/style_firsthit_matched.jsonl \
  --output-root /mnt/jfs/logs/illustrious/style_firsthit_matched_2see_sample_200 \
  --sample-count 200 \
  --seed 42 \
  --jpg-quality 75 \
  --workers 32

python3 /data/benchmark_metrics/lora_pipeline/tools/export_jsonl_sampled_image_groups.py \
  --jsonl /data/benchmark_metrics/logs/triplet_content_firsthit_judge_0325_0.5_2_0402_perimage/content_firsthit_matched.jsonl  \
  --output-root /mnt/jfs/logs/flux_triplet_content_firsthit_judge_0325_0.5_2_0402_perimage \
  --sample-count 200 \
  --seed 42 \
  --jpg-quality 75 \
  --workers 32

python3 /data/benchmark_metrics/lora_pipeline/tools/export_jsonl_sampled_image_groups.py \
  --jsonl /mnt/jfs/logs/qwen_triplet_content_firsthit_judge_0325_0.5_2_0403_perimage/content_firsthit_matched.jsonl \
  --output-root /mnt/jfs/logs/qwen_triplet_content_firsthit_judge_0325_0.5_2_0403_perimage_2see \
  --sample-count 200 \
  --seed 42 \
  --jpg-quality 75 \
  --workers 32

python3 /data/benchmark_metrics/lora_pipeline/tools/export_jsonl_sampled_image_groups.py \
  --jsonl /mnt/jfs/logs/illustrious_similar_people_binary_judge_20260410/illustrious_content_one_lora/similar_people_true.jsonl \
  --output-root /mnt/jfs/logs/illustrious_similar_people_binary_judge_20260410/illustrious_content_one_lora2see_2person \
  --sample-count 100 \
  --seed 42 \
  --jpg-quality 75 \
  --workers 32

python3 /data/benchmark_metrics/lora_pipeline/tools/export_jsonl_sampled_image_groups.py \
  --jsonl /data/benchmark_metrics/logs/triplet_jsonl/flux_style_content_intersection_by_model_combo_content_0325_0.5_1_0402_perimage.jsonl \
  --output-root /mnt/jfs/logs/illustrious_similar_people_binary_judge_20260410/flux_style_content_intersection_by_model_combo_content_0325_0.5_1_0402_perimage2see \
  --sample-count 400 \
  --seed 42 \
  --jpg-quality 75 \
  --workers 32
"""
import argparse
import hashlib
import io
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageOps

try:
    from megfile.smart import smart_exists, smart_open as mopen
except Exception:
    smart_exists = None
    mopen = None


@dataclass
class SampleRecord:
    line_number: int
    item_index: int
    key_path: str
    value_paths: List[str]


@dataclass
class CopyJob:
    sample_index: int
    slot_name: str
    src: str
    dst: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="输入 JSONL 路径")
    p.add_argument("--output-root", required=True, help="导出目录")
    p.add_argument("--sample-count", type=int, default=100, help="随机采样数量，0 表示导出全部可用样本")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--jpg-quality", type=int, default=80, help="JPEG 质量，范围 1~100")
    p.add_argument("--workers", type=int, default=16, help="并发线程数")
    p.add_argument("--overwrite", action="store_true", help="若目标文件已存在则覆盖")
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="每处理多少张图片打印一次进度，0 表示不打印中间进度",
    )
    return p.parse_args()


def sanitize_stem(text: str, max_len: int = 48) -> str:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    stem = stem.strip("._-")
    if not stem:
        stem = "sample"
    return stem[:max_len]


def normalize_value_paths(value: Any) -> List[str]:
    if isinstance(value, str):
        path = value.strip()
        return [path] if path else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            path = item.strip()
            if path:
                out.append(path)
        return out
    return []


def is_image_name(path: str) -> bool:
    lower = str(path or "").lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def is_remote_path(path: str) -> bool:
    s = str(path or "")
    return s.startswith("s3://") or s.startswith("oss://")


def path_exists(path: str) -> bool:
    if not isinstance(path, str):
        return False
    p = path.strip()
    if not p:
        return False
    if is_remote_path(p):
        if smart_exists is None:
            return False
        try:
            return bool(smart_exists(p))
        except Exception:
            return False
    return os.path.isfile(p)


def read_image_bytes(path: str) -> bytes:
    if is_remote_path(path):
        if mopen is None:
            raise RuntimeError(f"remote path requires megfile: {path}")
        with mopen(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def has_copyable_key_image(record: "SampleRecord") -> bool:
    return is_image_name(record.key_path) and path_exists(record.key_path)


def load_records(jsonl_path: str) -> Tuple[List[SampleRecord], Dict[str, int]]:
    path = Path(jsonl_path)
    if not path.is_file():
        raise RuntimeError(f"jsonl 不存在: {jsonl_path}")

    records: List[SampleRecord] = []
    stats = {
        "total_lines": 0,
        "bad_json_lines": 0,
        "non_dict_lines": 0,
        "invalid_items": 0,
        "empty_value_items": 0,
        "valid_items": 0,
    }

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = (raw_line or "").strip()
            if not line:
                continue
            stats["total_lines"] += 1
            try:
                obj = json.loads(line)
            except Exception:
                stats["bad_json_lines"] += 1
                continue
            if not isinstance(obj, dict):
                stats["non_dict_lines"] += 1
                continue

            for item_index, (key, value) in enumerate(obj.items(), start=1):
                if not isinstance(key, str):
                    stats["invalid_items"] += 1
                    continue
                key_path = key.strip()
                if not key_path:
                    stats["invalid_items"] += 1
                    continue
                value_paths = normalize_value_paths(value)
                if not value_paths:
                    stats["empty_value_items"] += 1
                    continue
                records.append(
                    SampleRecord(
                        line_number=line_number,
                        item_index=item_index,
                        key_path=key_path,
                        value_paths=value_paths,
                    )
                )
                stats["valid_items"] += 1

    return records, stats


def all_paths_exist(record: SampleRecord) -> bool:
    if not record.value_paths:
        return False
    return all(path_exists(path) for path in record.value_paths)


def select_records(records: List[SampleRecord], sample_count: int, seed: int) -> Tuple[List[SampleRecord], Dict[str, int]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    target_count = len(shuffled) if int(sample_count) <= 0 else int(sample_count)
    selected: List[SampleRecord] = []
    skipped_missing = 0

    for record in shuffled:
        if not all_paths_exist(record):
            skipped_missing += 1
            continue
        selected.append(record)
        if len(selected) >= target_count:
            break

    stats = {
        "requested": target_count,
        "selected": len(selected),
        "skipped_missing": skipped_missing,
    }
    return selected, stats


def make_basename(record: SampleRecord, sample_index: int) -> str:
    raw_stem = Path(record.key_path).stem
    safe_stem = sanitize_stem(raw_stem)
    digest_src = json.dumps(
        {
            "line_number": record.line_number,
            "item_index": record.item_index,
            "key_path": record.key_path,
            "value_paths": record.value_paths,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:10]
    return f"{sample_index:06d}__{safe_stem}__{digest}.jpg"


def ensure_output_dirs(output_root: Path, max_value_count: int, include_key_dir: bool):
    if include_key_dir:
        (output_root / "key").mkdir(parents=True, exist_ok=True)
    for i in range(1, max_value_count + 1):
        (output_root / f"value{i}").mkdir(parents=True, exist_ok=True)


def convert_to_jpg(src: str, dst: str, jpg_quality: int, overwrite: bool) -> str:
    if os.path.exists(dst) and not overwrite:
        return "skipped"

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    raw = read_image_bytes(src)
    with Image.open(io.BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img)
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        background.save(dst, format="JPEG", quality=int(jpg_quality), optimize=True)
    return "written"


def run_copy_job(job: CopyJob, jpg_quality: int, overwrite: bool) -> Tuple[bool, str]:
    try:
        status = convert_to_jpg(
            src=job.src,
            dst=job.dst,
            jpg_quality=jpg_quality,
            overwrite=overwrite,
        )
        return True, status
    except Exception as e:
        return False, str(e)


def build_manifest_and_jobs(output_root: Path, records: List[SampleRecord]) -> Tuple[List[Dict[str, Any]], List[CopyJob], bool]:
    manifest_rows: List[Dict[str, Any]] = []
    jobs: List[CopyJob] = []
    include_key_dir = False

    for sample_index, record in enumerate(records, start=1):
        basename = make_basename(record=record, sample_index=sample_index)
        copy_key = has_copyable_key_image(record)
        key_dst = str(output_root / "key" / basename) if copy_key else ""
        value_dsts = [str(output_root / f"value{i}" / basename) for i in range(1, len(record.value_paths) + 1)]

        manifest_rows.append(
            {
                "sample_index": sample_index,
                "basename": basename,
                "line_number": record.line_number,
                "item_index": record.item_index,
                "key_src": record.key_path,
                "key_dst": key_dst,
                "copy_key": copy_key,
                "value_srcs": list(record.value_paths),
                "value_dsts": value_dsts,
            }
        )

        if copy_key:
            include_key_dir = True
            jobs.append(
                CopyJob(
                    sample_index=sample_index,
                    slot_name="key",
                    src=record.key_path,
                    dst=key_dst,
                )
            )
        for idx, value_path in enumerate(record.value_paths, start=1):
            jobs.append(
                CopyJob(
                    sample_index=sample_index,
                    slot_name=f"value{idx}",
                    src=value_path,
                    dst=str(output_root / f"value{idx}" / basename),
                )
            )

    return manifest_rows, jobs, include_key_dir


def write_manifest(manifest_path: Path, rows: List[Dict[str, Any]]):
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    if int(args.jpg_quality) < 1 or int(args.jpg_quality) > 100:
        raise RuntimeError("jpg-quality 必须在 1~100")
    if int(args.workers) <= 0:
        raise RuntimeError("workers 必须 > 0")
    if int(args.sample_count) < 0:
        raise RuntimeError("sample-count 不能 < 0")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records, load_stats = load_records(args.jsonl)
    selected_records, select_stats = select_records(
        records=records,
        sample_count=int(args.sample_count),
        seed=int(args.seed),
    )
    if not selected_records:
        raise RuntimeError("没有可导出的有效样本，请检查 jsonl 路径、图片路径或 sample-count")

    max_value_count = max(len(record.value_paths) for record in selected_records)
    manifest_rows, jobs, include_key_dir = build_manifest_and_jobs(output_root=output_root, records=selected_records)
    ensure_output_dirs(
        output_root=output_root,
        max_value_count=max_value_count,
        include_key_dir=include_key_dir,
    )

    job_errors: List[Dict[str, Any]] = []
    written = 0
    skipped = 0
    failed = 0
    done = 0

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        future_to_job = {
            ex.submit(run_copy_job, job=job, jpg_quality=int(args.jpg_quality), overwrite=bool(args.overwrite)): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            ok, detail = future.result()
            done += 1
            if ok:
                if detail == "written":
                    written += 1
                else:
                    skipped += 1
            else:
                failed += 1
                job_errors.append(
                    {
                        "sample_index": job.sample_index,
                        "slot_name": job.slot_name,
                        "src": job.src,
                        "dst": job.dst,
                        "err": detail,
                    }
                )
            if int(args.progress_every) > 0 and done % int(args.progress_every) == 0:
                print(
                    f"progress images_done={done}/{len(jobs)} written={written} skipped={skipped} failed={failed}",
                    flush=True,
                )

    error_by_sample: Dict[int, List[Dict[str, Any]]] = {}
    for err in job_errors:
        error_by_sample.setdefault(int(err["sample_index"]), []).append(err)

    for row in manifest_rows:
        sample_errors = error_by_sample.get(int(row["sample_index"]), [])
        row["status"] = "failed" if sample_errors else "ok"
        row["errors"] = sample_errors

    write_manifest(output_root / "manifest.jsonl", manifest_rows)

    print(f"jsonl={args.jsonl}")
    print(f"output_root={output_root}")
    print(f"loaded_valid_items={load_stats['valid_items']}")
    print(f"skipped_empty_value_items={load_stats['empty_value_items']}")
    print(f"selected_samples={select_stats['selected']}")
    print(f"requested_samples={select_stats['requested']}")
    print(f"skipped_missing_candidates={select_stats['skipped_missing']}")
    print(f"include_key_dir={include_key_dir}")
    print(f"max_value_count={max_value_count}")
    print(f"images_total={len(jobs)}")
    print(f"images_written={written}")
    print(f"images_skipped={skipped}")
    print(f"images_failed={failed}")
    print(f"manifest={output_root / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()

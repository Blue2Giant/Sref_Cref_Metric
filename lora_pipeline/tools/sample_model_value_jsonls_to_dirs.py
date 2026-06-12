#!/usr/bin/env python3
"""
Randomly sample model->image-list jsonls into preview folders.

Expected input shapes:
1. {"<key>": ["<image_path_1>", "<image_path_2>", ...]}
2. {"<key>": "<image_path>"}

Output layout:
  output_root/
    <jsonl_stem>/
      value1/<key>.jpg
      value2/<key>.jpg
      ...
      manifest.jsonl
      summary.json
    run_summary.json
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
from typing import Any

from PIL import Image, ImageOps

try:
    from megfile import smart_exists, smart_open
except Exception:
    smart_exists = None
    smart_open = None


@dataclass
class SampleRecord:
    line_number: int
    item_index: int
    key: str
    value_paths: list[str]


@dataclass
class CopyJob:
    sample_key: str
    slot_name: str
    src: str
    dst: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample model->image-list jsonls into value1/value2 preview folders"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpg-quality", type=int, default=75)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--jsonl-glob", default="*.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_value_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            path = item.strip()
            if path:
                out.append(path)
        return out
    return []


def sanitize_key(key: str, max_len: int = 180) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", key.strip())
    safe = safe.strip("._-")
    if not safe:
        safe = "sample"
    return safe[:max_len]


def key_to_filename(key: str, used_names: dict[str, int]) -> str:
    safe = sanitize_key(key)
    count = used_names.get(safe, 0) + 1
    used_names[safe] = count
    if count == 1:
        return f"{safe}.jpg"
    return f"{safe}__dup{count}.jpg"


def read_path_bytes(path: str) -> bytes:
    if smart_open is not None:
        with smart_open(path, "rb") as f:
            return f.read()
    if str(path).startswith("s3://"):
        raise RuntimeError("megfile is required for s3 paths")
    with open(path, "rb") as f:
        return f.read()


def path_exists(path: str) -> bool:
    if smart_exists is not None:
        try:
            return bool(smart_exists(path))
        except Exception:
            return False
    if str(path).startswith("s3://"):
        return False
    return os.path.isfile(path)


def convert_to_jpg(src: str, dst: str, jpg_quality: int, overwrite: bool) -> str:
    if os.path.exists(dst) and not overwrite:
        return "skipped"

    raw = read_path_bytes(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Image.open(io.BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img)
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        background.save(dst, format="JPEG", quality=int(jpg_quality), optimize=True)
    return "written"


def run_copy_job(job: CopyJob, jpg_quality: int, overwrite: bool) -> tuple[bool, str]:
    try:
        status = convert_to_jpg(
            src=job.src,
            dst=job.dst,
            jpg_quality=jpg_quality,
            overwrite=overwrite,
        )
        return True, status
    except Exception as exc:
        return False, str(exc)


def load_records(jsonl_path: Path) -> tuple[list[SampleRecord], dict[str, int]]:
    records: list[SampleRecord] = []
    stats = {
        "total_lines": 0,
        "bad_json_lines": 0,
        "non_dict_lines": 0,
        "invalid_items": 0,
        "valid_items": 0,
        "empty_value_items": 0,
    }

    with jsonl_path.open("r", encoding="utf-8") as fin:
        for line_number, raw_line in enumerate(fin, start=1):
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
                key = key.strip()
                if not key:
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
                        key=key,
                        value_paths=value_paths,
                    )
                )
                stats["valid_items"] += 1

    return records, stats


def compute_file_seed(seed: int, file_name: str) -> int:
    digest = hashlib.sha1(f"{seed}:{file_name}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def select_records(records: list[SampleRecord], sample_count: int, seed: int) -> tuple[list[SampleRecord], dict[str, int]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    target_count = len(shuffled) if int(sample_count) <= 0 else min(int(sample_count), len(shuffled))
    selected: list[SampleRecord] = []
    skipped_missing = 0

    for record in shuffled:
        if not all(path_exists(path) for path in record.value_paths):
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


def ensure_output_dirs(output_root: Path, max_value_count: int):
    output_root.mkdir(parents=True, exist_ok=True)
    for idx in range(1, max_value_count + 1):
        (output_root / f"value{idx}").mkdir(parents=True, exist_ok=True)


def build_manifest_and_jobs(
    output_root: Path, records: list[SampleRecord]
) -> tuple[list[dict[str, Any]], list[CopyJob], int]:
    manifest_rows: list[dict[str, Any]] = []
    jobs: list[CopyJob] = []
    used_names: dict[str, int] = {}
    max_value_count = 0

    for record in records:
        max_value_count = max(max_value_count, len(record.value_paths))
        basename = key_to_filename(record.key, used_names=used_names)
        value_dsts = []
        for idx, value_path in enumerate(record.value_paths, start=1):
            dst = str(output_root / f"value{idx}" / basename)
            value_dsts.append(dst)
            jobs.append(
                CopyJob(
                    sample_key=record.key,
                    slot_name=f"value{idx}",
                    src=value_path,
                    dst=dst,
                )
            )

        manifest_rows.append(
            {
                "key": record.key,
                "basename": basename,
                "line_number": record.line_number,
                "item_index": record.item_index,
                "value_srcs": list(record.value_paths),
                "value_dsts": value_dsts,
            }
        )

    return manifest_rows, jobs, max_value_count


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest(path: Path, rows: list[dict[str, Any]]):
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_one_jsonl(
    jsonl_path: Path,
    output_root: Path,
    sample_count: int,
    seed: int,
    jpg_quality: int,
    workers: int,
    progress_every: int,
    overwrite: bool,
) -> dict[str, Any]:
    print(f"[START] jsonl={jsonl_path}", flush=True)
    records, load_stats = load_records(jsonl_path)
    selected_records, select_stats = select_records(
        records=records,
        sample_count=sample_count,
        seed=compute_file_seed(seed=seed, file_name=jsonl_path.name),
    )

    if not selected_records:
        summary = {
            "jsonl": str(jsonl_path),
            "output_dir": str(output_root),
            "status": "no_selected_records",
            **load_stats,
            **select_stats,
        }
        write_json(output_root / "summary.json", summary)
        print(f"[DONE] jsonl={jsonl_path} status=no_selected_records", flush=True)
        return summary

    manifest_rows, jobs, max_value_count = build_manifest_and_jobs(
        output_root=output_root, records=selected_records
    )
    ensure_output_dirs(output_root=output_root, max_value_count=max_value_count)

    written = 0
    skipped = 0
    failed = 0
    done = 0
    job_errors: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        future_to_job = {
            ex.submit(
                run_copy_job,
                job=job,
                jpg_quality=int(jpg_quality),
                overwrite=bool(overwrite),
            ): job
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
                        "key": job.sample_key,
                        "slot_name": job.slot_name,
                        "src": job.src,
                        "dst": job.dst,
                        "err": detail,
                    }
                )
            if int(progress_every) > 0 and done % int(progress_every) == 0:
                print(
                    f"[PROGRESS] jsonl={jsonl_path.name} images_done={done}/{len(jobs)} written={written} skipped={skipped} failed={failed}",
                    flush=True,
                )

    error_by_key: dict[str, list[dict[str, Any]]] = {}
    for err in job_errors:
        error_by_key.setdefault(str(err["key"]), []).append(err)

    for row in manifest_rows:
        row_errors = error_by_key.get(str(row["key"]), [])
        row["status"] = "failed" if row_errors else "ok"
        row["errors"] = row_errors

    manifest_path = output_root / "manifest.jsonl"
    write_manifest(manifest_path, manifest_rows)

    summary = {
        "jsonl": str(jsonl_path),
        "output_dir": str(output_root),
        "status": "ok",
        **load_stats,
        **select_stats,
        "max_value_count": max_value_count,
        "images_total": len(jobs),
        "images_written": written,
        "images_skipped": skipped,
        "images_failed": failed,
        "manifest": str(manifest_path),
    }
    write_json(output_root / "summary.json", summary)
    print(
        f"[DONE] jsonl={jsonl_path.name} selected={select_stats['selected']} images_total={len(jobs)} failed={failed} output_dir={output_root}",
        flush=True,
    )
    return summary


def main():
    args = parse_args()

    if int(args.sample_count) < 0:
        raise RuntimeError("sample-count must be >= 0")
    if int(args.workers) <= 0:
        raise RuntimeError("workers must be > 0")
    if int(args.jpg_quality) < 1 or int(args.jpg_quality) > 100:
        raise RuntimeError("jpg-quality must be in 1..100")

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    if not input_dir.is_dir():
        raise RuntimeError(f"input dir not found: {input_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    jsonl_paths = sorted(p for p in input_dir.glob(args.jsonl_glob) if p.is_file())
    if not jsonl_paths:
        raise RuntimeError(f"no jsonl found under {input_dir} with glob {args.jsonl_glob}")

    run_summaries: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths:
        per_jsonl_output = output_root / jsonl_path.stem
        summary = process_one_jsonl(
            jsonl_path=jsonl_path,
            output_root=per_jsonl_output,
            sample_count=int(args.sample_count),
            seed=int(args.seed),
            jpg_quality=int(args.jpg_quality),
            workers=int(args.workers),
            progress_every=int(args.progress_every),
            overwrite=bool(args.overwrite),
        )
        run_summaries.append(summary)

    run_summary = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "jsonl_count": len(jsonl_paths),
        "sample_count": int(args.sample_count),
        "seed": int(args.seed),
        "jpg_quality": int(args.jpg_quality),
        "workers": int(args.workers),
        "summaries": run_summaries,
    }
    write_json(output_root / "run_summary.json", run_summary)

    print(f"input_dir={input_dir}")
    print(f"output_root={output_root}")
    print(f"jsonl_count={len(jsonl_paths)}")
    print(f"run_summary={output_root / 'run_summary.json'}")


if __name__ == "__main__":
    main()

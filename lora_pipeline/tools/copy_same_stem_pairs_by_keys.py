#!/usr/bin/env python3
"""
Copy same-stem files from multiple subdirectories, optionally filtered by a key txt.

Typical usage:
python3 /data/benchmark_metrics/lora_pipeline/tools/copy_same_stem_pairs_by_keys.py \
  --src-root /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content \
  --dst-root /data/benchmark_metrics/logs/sample_800_cref_sref_200_content_analysis_subset \
  --key-txt /data/benchmark_metrics/insight/key_folder/qwen/analysis_key.txt \
  --subdir cref \
  --subdir sref \
  --subdir qwen-edit

If --key-txt is omitted, the script copies all stems that exist in every selected subdir.
"""

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_EXTS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
]
EXT_PRIORITY = {ext: idx for idx, ext in enumerate(DEFAULT_EXTS)}


@dataclass
class KeyRecord:
    key: str
    status: str
    missing_subdirs: List[str]
    selected_sources: Dict[str, str]
    destination_files: Dict[str, str]
    duplicate_candidates: Dict[str, List[str]]
    copy_status: Dict[str, str]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy same-stem files from multiple subdirs. If --key-txt is omitted, copy all common stems."
    )
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--key-txt", default="")
    parser.add_argument("--subdir", action="append", default=[])
    parser.add_argument("--subdir-file", default="")
    parser.add_argument("--allow-any-ext", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-name", default="copy_manifest.json")
    return parser.parse_args()


def load_subdirs(args) -> List[str]:
    seen = set()
    out: List[str] = []

    def add_one(text: str):
        item = str(text).strip().strip("/")
        if item and item not in seen:
            seen.add(item)
            out.append(item)

    for item in args.subdir:
        add_one(item)

    if args.subdir_file:
        path = Path(args.subdir_file)
        if not path.is_file():
            raise RuntimeError(f"subdir-file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            add_one(line)

    if not out:
        raise RuntimeError("at least one --subdir or --subdir-file is required")
    return out


def load_keys(path: Path) -> Tuple[List[str], Dict[str, int]]:
    if not path.is_file():
        raise RuntimeError(f"key txt not found: {path}")

    raw_keys: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        key = line.strip()
        if not key or key.startswith("#"):
            continue
        raw_keys.append(key)

    unique_keys: List[str] = []
    seen = set()
    duplicate_count = 0
    for key in raw_keys:
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_keys.append(key)

    stats = {
        "raw_key_count": len(raw_keys),
        "unique_key_count": len(unique_keys),
        "duplicate_key_count": duplicate_count,
    }
    return unique_keys, stats


def collect_common_keys(subdir_indexes: Dict[str, Dict[str, List[Path]]]) -> Tuple[List[str], Dict[str, int]]:
    key_sets = [set(index.keys()) for index in subdir_indexes.values()]
    if not key_sets:
        common_keys: List[str] = []
    else:
        common_keys = sorted(set.intersection(*key_sets))

    stats = {
        "raw_key_count": len(common_keys),
        "unique_key_count": len(common_keys),
        "duplicate_key_count": 0,
    }
    return common_keys, stats


def scan_subdir(root: Path, allow_any_ext: bool) -> Dict[str, List[Path]]:
    stem_to_files: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        if not allow_any_ext and path.suffix.lower() not in EXT_PRIORITY:
            continue
        stem_to_files[path.stem].append(path)
    return stem_to_files


def choose_candidate(paths: List[Path]) -> Tuple[Path, List[Path]]:
    ordered = sorted(
        paths,
        key=lambda p: (EXT_PRIORITY.get(p.suffix.lower(), 999), p.name.lower()),
    )
    return ordered[0], ordered[1:]


def copy_one(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> str:
    if dst.exists() and not overwrite:
        return "skipped_existing"
    if dry_run:
        return "dry_run"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    key_txt: Optional[Path] = Path(args.key_txt) if args.key_txt else None

    if not src_root.is_dir():
        raise RuntimeError(f"src-root not found: {src_root}")

    subdirs = load_subdirs(args)

    subdir_indexes: Dict[str, Dict[str, List[Path]]] = {}
    for subdir in subdirs:
        src_subdir = src_root / subdir
        if not src_subdir.is_dir():
            raise RuntimeError(f"subdir not found under src-root: {src_subdir}")
        subdir_indexes[subdir] = scan_subdir(src_subdir, allow_any_ext=bool(args.allow_any_ext))

    if key_txt is not None:
        keys, key_stats = load_keys(key_txt)
        key_mode = "from_txt"
    else:
        keys, key_stats = collect_common_keys(subdir_indexes)
        key_mode = "all_common_stems"

    records: List[KeyRecord] = []
    copied_file_count = 0
    skipped_existing_count = 0
    duplicate_pick_count = 0
    complete_key_count = 0
    missing_key_count = 0

    for key in keys:
        missing_subdirs: List[str] = []
        selected_sources: Dict[str, str] = {}
        destination_files: Dict[str, str] = {}
        duplicate_candidates: Dict[str, List[str]] = {}
        copy_status: Dict[str, str] = {}

        for subdir in subdirs:
            matches = subdir_indexes[subdir].get(key, [])
            if not matches:
                missing_subdirs.append(subdir)
                continue
            chosen, others = choose_candidate(matches)
            selected_sources[subdir] = str(chosen)
            if others:
                duplicate_pick_count += 1
                duplicate_candidates[subdir] = [str(item) for item in matches]

        if missing_subdirs:
            missing_key_count += 1
            records.append(
                KeyRecord(
                    key=key,
                    status="missing",
                    missing_subdirs=missing_subdirs,
                    selected_sources=selected_sources,
                    destination_files=destination_files,
                    duplicate_candidates=duplicate_candidates,
                    copy_status=copy_status,
                )
            )
            continue

        complete_key_count += 1
        for subdir in subdirs:
            src_path = Path(selected_sources[subdir])
            dst_path = dst_root / subdir / src_path.name
            destination_files[subdir] = str(dst_path)
            status = copy_one(
                src=src_path,
                dst=dst_path,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )
            copy_status[subdir] = status
            if status in {"copied", "dry_run"}:
                copied_file_count += 1
            elif status == "skipped_existing":
                skipped_existing_count += 1

        records.append(
            KeyRecord(
                key=key,
                status="complete",
                missing_subdirs=missing_subdirs,
                selected_sources=selected_sources,
                destination_files=destination_files,
                duplicate_candidates=duplicate_candidates,
                copy_status=copy_status,
            )
        )

    summary = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "key_txt": str(key_txt) if key_txt is not None else None,
        "key_mode": key_mode,
        "subdirs": subdirs,
        "allow_any_ext": bool(args.allow_any_ext),
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "raw_key_count": key_stats["raw_key_count"],
        "unique_key_count": key_stats["unique_key_count"],
        "duplicate_key_count": key_stats["duplicate_key_count"],
        "complete_key_count": complete_key_count,
        "missing_key_count": missing_key_count,
        "copied_or_planned_file_count": copied_file_count,
        "skipped_existing_file_count": skipped_existing_count,
        "duplicate_pick_count": duplicate_pick_count,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.dry_run:
        preview_records = [asdict(item) for item in records[:10]]
        print(json.dumps({"preview_records": preview_records}, ensure_ascii=False, indent=2), flush=True)
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dst_root / args.manifest_name
    copied_keys_path = dst_root / "copied_keys.txt"
    missing_keys_path = dst_root / "missing_keys.txt"

    manifest_payload = {
        "summary": summary,
        "records": [asdict(item) for item in records],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    copied_keys_path.write_text(
        "\n".join(item.key for item in records if item.status == "complete") + "\n",
        encoding="utf-8",
    )
    missing_keys_path.write_text(
        "\n".join(item.key for item in records if item.status == "missing") + "\n",
        encoding="utf-8",
    )

    print(f"manifest={manifest_path}", flush=True)
    print(f"copied_keys={copied_keys_path}", flush=True)
    print(f"missing_keys={missing_keys_path}", flush=True)


if __name__ == "__main__":
    main()

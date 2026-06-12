#!/usr/bin/env python3
"""Export one cref/sref/target triplet per style model id for paper figures.

Example:
  python3 /data/benchmark_metrics/lora_pipeline/tools/export_paper_triplet_samples.py \
    --output-root /mnt/jfs/loras_combine/paper_triplet_samples_qwen_flux_20260421 \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TRIPLET_ROOT = Path("/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls")

MODEL_CONFIG = {
    "flux": {
        "dual_filtered": TRIPLET_ROOT / "flux_dual_lora_style_content_filtered.jsonl",
        "content_one": TRIPLET_ROOT / "flux_content_one_lora.jsonl",
        "style_one": TRIPLET_ROOT / "flux_style_one_lora.jsonl",
    },
    "qwen": {
        "dual_filtered": TRIPLET_ROOT / "qwen_dual_lora_style_content_filtered.jsonl",
        "content_one": TRIPLET_ROOT / "qwen_content_one_lora.jsonl",
        "style_one": TRIPLET_ROOT / "qwen_style_one_lora.jsonl",
    },
}


@dataclass
class ExportRecord:
    base_model: str
    content_model_id: str
    style_model_id: str
    pair_key: str
    prompt_index: int
    caption: str
    full_prompt: str
    cref_src: str
    sref_src: str
    target_src: str
    cref_dst: str
    sref_dst: str
    target_dst: str


def record_from_dict(obj: dict) -> ExportRecord:
    return ExportRecord(
        base_model=str(obj["base_model"]),
        content_model_id=str(obj["content_model_id"]),
        style_model_id=str(obj["style_model_id"]),
        pair_key=str(obj["pair_key"]),
        prompt_index=int(obj["prompt_index"]),
        caption=str(obj["caption"]),
        full_prompt=str(obj["full_prompt"]),
        cref_src=str(obj["cref_src"]),
        sref_src=str(obj["sref_src"]),
        target_src=str(obj["target_src"]),
        cref_dst=str(obj["cref_dst"]),
        sref_dst=str(obj["sref_dst"]),
        target_dst=str(obj["target_dst"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-models",
        nargs="+",
        default=["flux", "qwen"],
        choices=sorted(MODEL_CONFIG),
        help="Base models to export.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/jfs/loras_combine/paper_triplet_samples_qwen_flux_20260421",
        help="Flat output directory for exported images and manifests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-base-model limit on selected style ids, for debugging.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Copy workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan selections without copying files.",
    )
    parser.add_argument(
        "--manifest-jsonl",
        default="",
        help="Reuse an existing manifest.jsonl and only perform destination rewrite/copy.",
    )
    return parser.parse_args()


def read_jsonl_map(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            for key, value in obj.items():
                if isinstance(key, str) and isinstance(value, list):
                    out[key] = [str(x) for x in value if isinstance(x, str) and str(x).strip()]
    return out


def read_manifest_records(path: Path) -> List[ExportRecord]:
    records: List[ExportRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            records.append(record_from_dict(json.loads(s)))
    return records


def prefer_local_paths(paths: Sequence[str]) -> List[str]:
    local = [p for p in paths if p.startswith("/mnt/")]
    remote = [p for p in paths if not p.startswith("/mnt/")]
    return local + remote


def existing_paths(paths: Sequence[str], exists_cache: Dict[str, bool]) -> List[str]:
    out: List[str] = []
    seen = set()
    for path in prefer_local_paths(paths):
        if path in seen:
            continue
        seen.add(path)
        if path_exists(path, exists_cache):
            out.append(path)
    return out


def path_exists(path: str, exists_cache: Dict[str, bool]) -> bool:
    hit = exists_cache.get(path)
    if hit is not None:
        return hit
    hit = os.path.exists(path)
    exists_cache[path] = hit
    return hit


def first_existing_path(paths: Sequence[str], exists_cache: Dict[str, bool]) -> Optional[str]:
    for path in prefer_local_paths(paths):
        if path.startswith("/mnt/") and path_exists(path, exists_cache):
            return path
    for path in prefer_local_paths(paths):
        if path_exists(path, exists_cache):
            return path
    return None


def parse_pair_key(pair_key: str) -> Tuple[str, str]:
    parts = pair_key.split("__", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"bad pair key: {pair_key}")
    return parts[0], parts[1]


def parse_prompt_index(image_path: str) -> Optional[int]:
    stem = Path(image_path).stem
    prefix = stem.split("_", 1)[0]
    if not prefix.isdigit():
        return None
    return int(prefix)


def load_prompt_payload(prompt_path: Path, prompt_cache: Dict[str, Optional[dict]]) -> Optional[dict]:
    cache_key = str(prompt_path)
    if cache_key in prompt_cache:
        return prompt_cache[cache_key]
    if not prompt_path.exists():
        prompt_cache[cache_key] = None
        return None
    with prompt_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    prompt_cache[cache_key] = payload
    return payload


def prompt_for_target(
    target_path: str,
    prompt_cache: Dict[str, Optional[dict]],
) -> Optional[Tuple[int, str, str]]:
    prompt_index = parse_prompt_index(target_path)
    if prompt_index is None:
        return None
    pair_dir = Path(target_path).parents[1]
    payload = load_prompt_payload(pair_dir / "selected_prompts_final.json", prompt_cache)
    if not isinstance(payload, dict):
        return None
    selected_prompts = payload.get("selected_prompts")
    base_prompts = payload.get("selected_base_prompts")
    if not isinstance(selected_prompts, list):
        return None
    if prompt_index >= len(selected_prompts):
        return None
    full_prompt = str(selected_prompts[prompt_index]).strip()
    caption = ""
    if isinstance(base_prompts, list) and prompt_index < len(base_prompts):
        caption = str(base_prompts[prompt_index]).strip()
    if not caption:
        caption = full_prompt
    if not full_prompt:
        return None
    return prompt_index, caption, full_prompt


def slugify_prompt(text: str, max_len: int = 96) -> str:
    normalized = text.lower().strip()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        normalized = "prompt"
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    budget = max(8, max_len - 9)
    normalized = normalized[:budget].strip("_")
    return f"{normalized}_{digest}"


def build_destination_paths(
    output_root: Path,
    base_model: str,
    style_model_id: str,
    prompt_text: str,
    ext: str,
) -> Dict[str, Path]:
    slug = slugify_prompt(prompt_text)
    prefix = f"{base_model}_{style_model_id}_{slug}"
    return {
        "cref": output_root / f"{prefix}_cref{ext}",
        "sref": output_root / f"{prefix}_sref{ext}",
        "target": output_root / f"{prefix}_target{ext}",
    }


def select_one_per_style(
    base_model: str,
    dual_filtered: Dict[str, List[str]],
    content_one: Dict[str, List[str]],
    style_one: Dict[str, List[str]],
    limit: int = 0,
) -> Tuple[List[ExportRecord], Dict[str, int]]:
    exists_cache: Dict[str, bool] = {}
    prompt_cache: Dict[str, Optional[dict]] = {}
    content_paths_by_model = {
        model_id: existing_paths(paths, exists_cache)
        for model_id, paths in content_one.items()
    }
    best_style = {
        model_id: first_existing_path(paths, exists_cache)
        for model_id, paths in style_one.items()
    }
    by_style: Dict[str, List[str]] = {}
    for pair_key in sorted(dual_filtered):
        try:
            _, style_model_id = parse_pair_key(pair_key)
        except ValueError:
            continue
        by_style.setdefault(style_model_id, []).append(pair_key)

    stats = {
        "styles_total": len(by_style),
        "styles_selected": 0,
        "skip_no_content": 0,
        "skip_no_style": 0,
        "skip_no_target": 0,
        "skip_no_prompt": 0,
        "skip_bad_pair": 0,
    }
    selected: List[ExportRecord] = []
    content_use_count: Dict[str, int] = {}
    content_image_cursor: Dict[str, int] = {}

    style_option_counts = {
        style_model_id: len(
            {
                pair_key.split("__", 1)[0]
                for pair_key in pair_keys
                if "__" in pair_key
            }
        )
        for style_model_id, pair_keys in by_style.items()
    }
    sorted_style_ids = sorted(
        by_style,
        key=lambda style_model_id: (style_option_counts.get(style_model_id, 0), style_model_id),
    )
    for style_idx, style_model_id in enumerate(sorted_style_ids, start=1):
        if limit and len(selected) >= limit:
            break
        chosen: Optional[ExportRecord] = None
        valid_candidates = []
        for pair_key in by_style[style_model_id]:
            try:
                content_model_id, style_id_check = parse_pair_key(pair_key)
            except ValueError:
                stats["skip_bad_pair"] += 1
                continue
            if style_id_check != style_model_id:
                stats["skip_bad_pair"] += 1
                continue

            content_paths = content_paths_by_model.get(content_model_id) or []
            if not content_paths:
                stats["skip_no_content"] += 1
                continue
            sref_src = best_style.get(style_model_id)
            if not sref_src:
                stats["skip_no_style"] += 1
                continue

            target_src: Optional[str] = None
            prompt_info: Optional[Tuple[int, str, str]] = None
            for candidate in prefer_local_paths(dual_filtered.get(pair_key, [])):
                if not path_exists(candidate, exists_cache):
                    continue
                prompt_info = prompt_for_target(candidate, prompt_cache)
                if prompt_info is None:
                    continue
                target_src = candidate
                break

            if not target_src:
                if dual_filtered.get(pair_key):
                    stats["skip_no_prompt"] += 1
                else:
                    stats["skip_no_target"] += 1
                continue
            if prompt_info is None:
                stats["skip_no_prompt"] += 1
                continue

            prompt_index, caption, full_prompt = prompt_info
            ext = Path(target_src).suffix or ".png"
            dst_paths = build_destination_paths(
                output_root=Path(""),
                base_model=base_model,
                style_model_id=style_model_id,
                prompt_text=caption,
                ext=ext,
            )
            valid_candidates.append(
                {
                    "base_model": base_model,
                    "content_model_id": content_model_id,
                    "style_model_id": style_model_id,
                    "pair_key": pair_key,
                    "prompt_index": prompt_index,
                    "caption": caption,
                    "full_prompt": full_prompt,
                    "content_paths": content_paths,
                    "sref_src": sref_src,
                    "target_src": target_src,
                    "cref_dst": str(dst_paths["cref"]),
                    "sref_dst": str(dst_paths["sref"]),
                    "target_dst": str(dst_paths["target"]),
                }
            )

        if valid_candidates:
            valid_candidates.sort(
                key=lambda candidate: (
                    content_use_count.get(candidate["content_model_id"], 0),
                    candidate["content_model_id"],
                    candidate["pair_key"],
                )
            )
            winner = valid_candidates[0]
            content_model_id = str(winner["content_model_id"])
            content_paths = winner["content_paths"]
            image_index = content_image_cursor.get(content_model_id, 0) % len(content_paths)
            cref_src = str(content_paths[image_index])
            content_image_cursor[content_model_id] = content_image_cursor.get(content_model_id, 0) + 1
            content_use_count[content_model_id] = content_use_count.get(content_model_id, 0) + 1
            chosen = ExportRecord(
                base_model=str(winner["base_model"]),
                content_model_id=content_model_id,
                style_model_id=str(winner["style_model_id"]),
                pair_key=str(winner["pair_key"]),
                prompt_index=int(winner["prompt_index"]),
                caption=str(winner["caption"]),
                full_prompt=str(winner["full_prompt"]),
                cref_src=cref_src,
                sref_src=str(winner["sref_src"]),
                target_src=str(winner["target_src"]),
                cref_dst=str(winner["cref_dst"]),
                sref_dst=str(winner["sref_dst"]),
                target_dst=str(winner["target_dst"]),
            )

        if chosen is not None:
            selected.append(chosen)
            stats["styles_selected"] += 1
        if style_idx % 200 == 0 or style_idx == len(sorted_style_ids):
            print(
                f"[{base_model}] scanned {style_idx}/{len(sorted_style_ids)} styles, "
                f"selected={stats['styles_selected']} unique_content={len(content_use_count)}",
                flush=True,
            )

    stats["unique_content_models"] = len(content_use_count)
    stats["reused_content_records"] = sum(max(0, count - 1) for count in content_use_count.values())
    return selected, stats


def rewrite_destinations(records: Iterable[ExportRecord], output_root: Path) -> List[ExportRecord]:
    out: List[ExportRecord] = []
    for record in records:
        ext = Path(record.target_src).suffix or ".png"
        dst_paths = build_destination_paths(
            output_root=output_root,
            base_model=record.base_model,
            style_model_id=record.style_model_id,
            prompt_text=record.caption,
            ext=ext,
        )
        out.append(
            ExportRecord(
                **{
                    **asdict(record),
                    "cref_dst": str(dst_paths["cref"]),
                    "sref_dst": str(dst_paths["sref"]),
                    "target_dst": str(dst_paths["target"]),
                }
            )
        )
    return out


def copy_one(src: str, dst: str, overwrite: bool) -> Tuple[str, bool]:
    if not overwrite and os.path.exists(dst):
        return dst, False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst, True


def write_manifests(output_root: Path, records: Sequence[ExportRecord], stats: dict) -> None:
    manifest_jsonl = output_root / "manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    manifest_csv = output_root / "manifest.csv"
    fieldnames = list(asdict(records[0]).keys()) if records else list(ExportRecord.__annotations__.keys())
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_records: List[ExportRecord] = []
    all_stats: Dict[str, dict] = {}

    if args.manifest_jsonl:
        manifest_path = Path(args.manifest_jsonl)
        all_records = rewrite_destinations(read_manifest_records(manifest_path), output_root)
        all_stats["_manifest_reuse"] = {
            "manifest_jsonl": str(manifest_path),
            "records": len(all_records),
        }
        print(f"[manifest] loaded {len(all_records)} records from {manifest_path}", flush=True)
    else:
        for base_model in args.base_models:
            config = MODEL_CONFIG[base_model]
            dual_filtered = read_jsonl_map(config["dual_filtered"])
            content_one = read_jsonl_map(config["content_one"])
            style_one = read_jsonl_map(config["style_one"])
            records, stats = select_one_per_style(
                base_model=base_model,
                dual_filtered=dual_filtered,
                content_one=content_one,
                style_one=style_one,
                limit=args.limit,
            )
            records = rewrite_destinations(records, output_root)
            all_records.extend(records)
            all_stats[base_model] = stats
            print(
                f"[{base_model}] selected {stats['styles_selected']} / {stats['styles_total']} style ids",
                flush=True,
            )

    if args.dry_run:
        print(f"[dry-run] planned records: {len(all_records)}", flush=True)
        write_manifests(output_root, all_records, all_stats)
        return 0

    copy_jobs: List[Tuple[str, str]] = []
    for record in all_records:
        copy_jobs.extend(
            [
                (record.cref_src, record.cref_dst),
                (record.sref_src, record.sref_dst),
                (record.target_src, record.target_dst),
            ]
        )

    copied = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(copy_one, src, dst, args.overwrite) for src, dst in copy_jobs]
        for future in as_completed(futures):
            _, did_copy = future.result()
            if did_copy:
                copied += 1
            else:
                skipped += 1

    all_stats["_copy"] = {
        "records": len(all_records),
        "files_total": len(copy_jobs),
        "files_copied": copied,
        "files_skipped_existing": skipped,
    }
    write_manifests(output_root, all_records, all_stats)
    print(
        f"[done] records={len(all_records)} files_total={len(copy_jobs)} copied={copied} skipped={skipped} output={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

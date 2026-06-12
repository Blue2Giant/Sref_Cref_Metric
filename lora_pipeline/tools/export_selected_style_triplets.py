#!/usr/bin/env python3
"""Plan or materialize triplet exports for selected style model ids.

Two-stage workflow:
1. plan
   Read style/content/dual jsonls, pick triplets, and write a manifest.
2. materialize
   Read the manifest on a machine that can access the image paths and copy
   them to the final output directory.

Example plan step:
  python3 /data/benchmark_metrics/lora_pipeline/tools/export_selected_style_triplets.py \
    --mode plan \
    --pair-source raw \
    --style-id-file flux=/data/benchmark_metrics/lora_pipeline/meta/model_ids/liked_flux_20260501.txt \
    --style-id-file qwen=/data/benchmark_metrics/lora_pipeline/meta/model_ids/liked_qwen_20260501.txt \
    --pairs-per-style 20 \
    --seed 20260501 \
    --output-root /mnt/jfs/liked_style_triplets_20260501 \
    --manifest-jsonl /tmp/liked_style_triplets_20260501.manifest.jsonl \
    --triplets-csv /tmp/liked_style_triplets_20260501.triplets.csv \
    --summary-json /tmp/liked_style_triplets_20260501.summary.json

Example materialize step:
  python3 /data/benchmark_metrics/lora_pipeline/tools/export_selected_style_triplets.py \
    --mode materialize \
    --manifest-jsonl /tmp/liked_style_triplets_20260501.manifest.jsonl \
    --summary-json /tmp/liked_style_triplets_20260501.materialize.summary.json \
    --workers 16 \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps

try:
    from megfile import smart_exists, smart_open
except Exception:
    smart_exists = None
    smart_open = None


TRIPLET_ROOT = Path("/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls")

MODEL_CONFIG = {
    "flux": {
        "dual_raw": TRIPLET_ROOT / "flux__dual_lora.jsonl",
        "dual_filtered": TRIPLET_ROOT / "flux_dual_lora_style_content_filtered.jsonl",
        "content_one": TRIPLET_ROOT / "flux_content_one_lora.jsonl",
        "style_one": TRIPLET_ROOT / "flux_style_one_lora.jsonl",
    },
    "qwen": {
        "dual_raw": TRIPLET_ROOT / "qwen_dual_lora.jsonl",
        "dual_filtered": TRIPLET_ROOT / "qwen_dual_lora_style_content_filtered.jsonl",
        "content_one": TRIPLET_ROOT / "qwen_content_one_lora.jsonl",
        "style_one": TRIPLET_ROOT / "qwen_style_one_lora.jsonl",
    },
    "illustrious": {
        "dual_raw": TRIPLET_ROOT / "illustrious_dual_lora.jsonl",
        "dual_filtered": TRIPLET_ROOT / "illustrious_dual_lora_style_content_filtered.jsonl",
        "content_one": TRIPLET_ROOT / "illustrious_content_one_lora.jsonl",
        "style_one": TRIPLET_ROOT / "illustrious_style_one_lora.jsonl",
    },
}


@dataclass(frozen=True)
class PairCandidate:
    base_model: str
    pair_key: str
    content_model_id: str
    style_model_id: str
    target_paths: Tuple[str, ...]


@dataclass
class TripletRecord:
    triplet_uuid: str
    base_model: str
    style_model_id: str
    content_model_id: str
    pair_key: str
    sample_index: int
    pair_cycle_index: int
    unique_pair_candidates: int
    style_source_path: str
    content_source_path: str
    target_source_path: str
    style_output_path: str
    content_output_path: str
    target_output_path: str


class CyclingSampler:
    """Cycle through items with deterministic reshuffle between rounds."""

    def __init__(self, items: Sequence, seed_token: str):
        self._items = list(items)
        self._seed_token = str(seed_token)
        self._round = 0
        self._index = 0
        self._order: List = []
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = list(self._items)
        random.Random(f"{self._seed_token}:{self._round}").shuffle(self._order)
        self._index = 0

    def next(self):
        if not self._order:
            raise RuntimeError("sampler has no items")
        if self._index >= len(self._order):
            self._round += 1
            self._reshuffle()
        item = self._order[self._index]
        self._index += 1
        return item, self._round


class PreferredPathSampler(CyclingSampler):
    """Deterministically shuffle local and remote paths separately, then prefer local."""

    def _reshuffle(self) -> None:
        local = [item for item in self._items if isinstance(item, str) and item.startswith("/mnt/")]
        remote = [item for item in self._items if item not in local]
        random.Random(f"{self._seed_token}:{self._round}:local").shuffle(local)
        random.Random(f"{self._seed_token}:{self._round}:remote").shuffle(remote)
        self._order = local + remote
        self._index = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("plan", "materialize"),
        required=True,
        help="plan: build manifest. materialize: copy images from manifest.",
    )
    parser.add_argument(
        "--pair-source",
        choices=("raw", "filtered"),
        default="raw",
        help="Which dual-lora jsonl to use during plan mode.",
    )
    parser.add_argument(
        "--style-id-file",
        action="append",
        default=[],
        help="Format: base_model=/abs/path/to/model_ids.txt . Can be repeated.",
    )
    parser.add_argument(
        "--pairs-per-style",
        type=int,
        default=20,
        help="Target number of triplets per requested style model id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260501,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Final image output root. Required in plan mode.",
    )
    parser.add_argument(
        "--manifest-jsonl",
        required=True,
        help="Manifest path. Written in plan mode, read in materialize mode.",
    )
    parser.add_argument(
        "--triplets-csv",
        default="",
        help="Optional CSV export for planned triplets.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary output path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Materialize worker count.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=92,
        help="JPEG quality used in materialize mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files in materialize mode.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_style_id_file(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            ids.extend(line.split())
    seen = set()
    out: List[str] = []
    for model_id in ids:
        if model_id not in seen:
            seen.add(model_id)
            out.append(model_id)
    return out


def load_requested_style_ids(specs: Sequence[str]) -> Dict[str, List[str]]:
    requested: Dict[str, List[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise RuntimeError(f"bad --style-id-file spec: {spec}")
        base_model, path_text = spec.split("=", 1)
        base_model = base_model.strip()
        if base_model not in MODEL_CONFIG:
            raise RuntimeError(f"unsupported base model in --style-id-file: {base_model}")
        path = Path(path_text.strip())
        if not path.is_file():
            raise RuntimeError(f"style id file does not exist: {path}")
        requested[base_model] = read_style_id_file(path)
    return requested


def normalize_value_paths(value) -> List[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item:
            out.append(item)
    return out


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def prefer_local_first(paths: Sequence[str]) -> List[str]:
    local = [p for p in paths if p.startswith("/mnt/")]
    remote = [p for p in paths if not p.startswith("/mnt/")]
    return dedupe_preserve(local + remote)


def prefer_local_only_if_available(paths: Sequence[str]) -> List[str]:
    local = [p for p in paths if p.startswith("/mnt/")]
    if local:
        return dedupe_preserve(local)
    return prefer_local_first(paths)


def read_jsonl_map(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            for key, value in obj.items():
                if not isinstance(key, str):
                    continue
                paths = normalize_value_paths(value)
                if paths:
                    out[key] = paths
    return out


def build_pair_index(base_model: str, pair_map: Dict[str, List[str]]) -> Dict[str, List[PairCandidate]]:
    out: Dict[str, List[PairCandidate]] = defaultdict(list)
    for pair_key, target_paths in pair_map.items():
        parts = pair_key.split("__", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        content_model_id, style_model_id = parts
        paths = tuple(prefer_local_first(target_paths))
        if not paths:
            continue
        out[style_model_id].append(
            PairCandidate(
                base_model=base_model,
                pair_key=pair_key,
                content_model_id=content_model_id,
                style_model_id=style_model_id,
                target_paths=paths,
            )
        )
    return out


def plan_triplets(
    requested_ids: Dict[str, List[str]],
    pair_source: str,
    pairs_per_style: int,
    seed: int,
    output_root: Path,
) -> Tuple[List[TripletRecord], dict]:
    all_records: List[TripletRecord] = []
    summary: dict = {
        "pair_source": pair_source,
        "pairs_per_style": int(pairs_per_style),
        "seed": int(seed),
        "output_root": str(output_root),
        "per_base_model": {},
    }

    global_content_samplers: Dict[Tuple[str, str], CyclingSampler] = {}
    global_target_samplers: Dict[Tuple[str, str], CyclingSampler] = {}

    for base_model, style_ids in requested_ids.items():
        config = MODEL_CONFIG[base_model]
        pair_key_name = "dual_raw" if pair_source == "raw" else "dual_filtered"
        pair_map = read_jsonl_map(config[pair_key_name])
        content_map = read_jsonl_map(config["content_one"])
        style_map = read_jsonl_map(config["style_one"])
        pair_index = build_pair_index(base_model=base_model, pair_map=pair_map)

        base_records: List[TripletRecord] = []
        base_summary = {
            "requested_style_ids": len(style_ids),
            "planned_triplets": 0,
            "style_ids_with_no_style_images": [],
            "style_ids_with_no_pair_candidates": [],
            "style_ids_with_unique_pair_count_lt_target": {},
            "style_ids_with_no_valid_pairs": [],
            "per_style": {},
        }

        for style_id in style_ids:
            style_paths = prefer_local_first(style_map.get(style_id, []))
            if not style_paths:
                base_summary["style_ids_with_no_style_images"].append(style_id)
                base_summary["per_style"][style_id] = {
                    "planned_triplets": 0,
                    "unique_pair_candidates": 0,
                    "reason": "missing_style_images",
                }
                continue

            all_candidates = pair_index.get(style_id, [])
            if not all_candidates:
                base_summary["style_ids_with_no_pair_candidates"].append(style_id)
                base_summary["per_style"][style_id] = {
                    "planned_triplets": 0,
                    "unique_pair_candidates": 0,
                    "reason": "missing_pair_candidates",
                }
                continue

            valid_candidates: List[PairCandidate] = []
            for candidate in all_candidates:
                content_paths = prefer_local_only_if_available(content_map.get(candidate.content_model_id, []))
                if not content_paths:
                    continue
                if not candidate.target_paths:
                    continue
                valid_candidates.append(candidate)

            if not valid_candidates:
                base_summary["style_ids_with_no_valid_pairs"].append(style_id)
                base_summary["per_style"][style_id] = {
                    "planned_triplets": 0,
                    "unique_pair_candidates": 0,
                    "reason": "missing_content_or_target_images",
                }
                continue

            if len(valid_candidates) < int(pairs_per_style):
                base_summary["style_ids_with_unique_pair_count_lt_target"][style_id] = len(valid_candidates)

            style_sampler = CyclingSampler(
                style_paths,
                seed_token=f"{seed}:{base_model}:{style_id}:style",
            )
            pair_sampler = CyclingSampler(
                valid_candidates,
                seed_token=f"{seed}:{base_model}:{style_id}:pairs",
            )

            planned_for_style = 0
            for sample_idx in range(1, int(pairs_per_style) + 1):
                candidate, pair_cycle_index = pair_sampler.next()
                style_src, _ = style_sampler.next()

                content_sampler_key = (base_model, candidate.content_model_id)
                if content_sampler_key not in global_content_samplers:
                    global_content_samplers[content_sampler_key] = PreferredPathSampler(
                        prefer_local_only_if_available(content_map[candidate.content_model_id]),
                        seed_token=f"{seed}:{base_model}:{candidate.content_model_id}:content",
                    )
                content_src, _ = global_content_samplers[content_sampler_key].next()

                target_sampler_key = (base_model, candidate.pair_key)
                if target_sampler_key not in global_target_samplers:
                    global_target_samplers[target_sampler_key] = PreferredPathSampler(
                        list(candidate.target_paths),
                        seed_token=f"{seed}:{base_model}:{candidate.pair_key}:target",
                    )
                target_src, _ = global_target_samplers[target_sampler_key].next()

                triplet_uuid = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{seed}:{base_model}:{style_id}:{sample_idx}:{candidate.pair_key}:{pair_cycle_index}",
                    )
                )
                style_dst = output_root / base_model / style_id / f"{triplet_uuid}+sref.jpg"
                content_dst = output_root / base_model / style_id / f"{triplet_uuid}+cref.jpg"
                target_dst = output_root / base_model / style_id / f"{triplet_uuid}+target.jpg"

                record = TripletRecord(
                    triplet_uuid=triplet_uuid,
                    base_model=base_model,
                    style_model_id=style_id,
                    content_model_id=candidate.content_model_id,
                    pair_key=candidate.pair_key,
                    sample_index=sample_idx,
                    pair_cycle_index=pair_cycle_index,
                    unique_pair_candidates=len(valid_candidates),
                    style_source_path=style_src,
                    content_source_path=content_src,
                    target_source_path=target_src,
                    style_output_path=str(style_dst),
                    content_output_path=str(content_dst),
                    target_output_path=str(target_dst),
                )
                base_records.append(record)
                all_records.append(record)
                planned_for_style += 1

            base_summary["per_style"][style_id] = {
                "planned_triplets": planned_for_style,
                "unique_pair_candidates": len(valid_candidates),
            }

        base_summary["planned_triplets"] = len(base_records)
        summary["per_base_model"][base_model] = base_summary

    summary["planned_triplets"] = len(all_records)
    summary["status_counts"] = {"planned": len(all_records)}
    return all_records, summary


def write_manifest(records: Sequence[TripletRecord], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_triplets_csv(records: Sequence[TripletRecord], path: Path) -> None:
    ensure_parent(path)
    fieldnames = [
        "triplet_uuid",
        "base_model",
        "style_model_id",
        "content_model_id",
        "pair_key",
        "sample_index",
        "pair_cycle_index",
        "unique_pair_candidates",
        "style_source_path",
        "content_source_path",
        "target_source_path",
        "style_output_path",
        "content_output_path",
        "target_output_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(path: Path) -> List[TripletRecord]:
    records: List[TripletRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            records.append(TripletRecord(**json.loads(line)))
    return records


def is_remote_path(path: str) -> bool:
    return str(path).startswith("s3://") or str(path).startswith("oss://")


def read_image_bytes(path: str) -> bytes:
    if is_remote_path(path):
        if smart_open is None:
            raise RuntimeError(f"remote source requires megfile: {path}")
        with smart_open(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def source_exists(path: str) -> bool:
    if is_remote_path(path):
        if smart_exists is None:
            return False
        try:
            return bool(smart_exists(path))
        except Exception:
            return False
    return os.path.isfile(path)


def save_as_jpg(src_path: str, dst_path: str, jpg_quality: int, overwrite: bool) -> str:
    if os.path.exists(dst_path) and not overwrite:
        return "skipped_existing"
    raw = read_image_bytes(src_path)
    with Image.open(io.BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img)
        rgba = img.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dst_path, format="JPEG", quality=int(jpg_quality), optimize=True)
    return "written"


def materialize_one(record: TripletRecord, jpg_quality: int, overwrite: bool) -> dict:
    outputs = [
        ("sref", record.style_source_path, record.style_output_path),
        ("cref", record.content_source_path, record.content_output_path),
        ("target", record.target_source_path, record.target_output_path),
    ]
    copy_status = {}
    for role, src, dst in outputs:
        if not source_exists(src):
            return {
                "triplet_uuid": record.triplet_uuid,
                "base_model": record.base_model,
                "style_model_id": record.style_model_id,
                "pair_key": record.pair_key,
                "status": "missing_source",
                "role": role,
                "source_path": src,
                "output_path": dst,
            }
        try:
            copy_status[role] = save_as_jpg(
                src_path=src,
                dst_path=dst,
                jpg_quality=jpg_quality,
                overwrite=overwrite,
            )
        except Exception as exc:
            return {
                "triplet_uuid": record.triplet_uuid,
                "base_model": record.base_model,
                "style_model_id": record.style_model_id,
                "pair_key": record.pair_key,
                "status": "copy_error",
                "role": role,
                "source_path": src,
                "output_path": dst,
                "error": repr(exc),
            }
    return {
        "triplet_uuid": record.triplet_uuid,
        "base_model": record.base_model,
        "style_model_id": record.style_model_id,
        "pair_key": record.pair_key,
        "status": "ok",
        "copy_status": copy_status,
    }


def materialize_triplets(
    records: Sequence[TripletRecord],
    jpg_quality: int,
    overwrite: bool,
    workers: int,
) -> dict:
    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [
            executor.submit(materialize_one, record, jpg_quality, overwrite)
            for record in records
        ]
        total = len(futures)
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == total:
                counts = Counter(item["status"] for item in results)
                print(
                    f"progress {index}/{total} {dict(counts)}",
                    flush=True,
                )

    status_counts = Counter(item["status"] for item in results)
    per_base_model: Dict[str, Counter] = defaultdict(Counter)
    for item in results:
        per_base_model[item["base_model"]][item["status"]] += 1

    return {
        "total_triplets": len(records),
        "status_counts": dict(status_counts),
        "per_base_model": {
            base_model: dict(counter)
            for base_model, counter in sorted(per_base_model.items())
        },
        "errors": [item for item in results if item["status"] != "ok"][:200],
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_jsonl)

    if args.mode == "plan":
        if not args.output_root:
            raise RuntimeError("--output-root is required in plan mode")
        requested_ids = load_requested_style_ids(args.style_id_file)
        output_root = Path(args.output_root)
        records, summary = plan_triplets(
            requested_ids=requested_ids,
            pair_source=args.pair_source,
            pairs_per_style=args.pairs_per_style,
            seed=args.seed,
            output_root=output_root,
        )
        write_manifest(records, manifest_path)
        if args.triplets_csv:
            write_triplets_csv(records, Path(args.triplets_csv))
        if args.summary_json:
            summary_path = Path(args.summary_json)
            ensure_parent(summary_path)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    records = read_manifest(manifest_path)
    summary = materialize_triplets(
        records=records,
        jpg_quality=args.jpg_quality,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json)
        ensure_parent(summary_path)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

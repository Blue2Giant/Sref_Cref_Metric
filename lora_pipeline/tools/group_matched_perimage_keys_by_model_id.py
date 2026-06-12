#!/usr/bin/env python3
"""
Group matched per-image jsonl keys by model_id extracted from image paths.

Input:
{"/path/or/s3://.../<model_id>/<subdir>/<image>.png": [...]}

Output:
{"<model_id>": ["/path/or/s3://.../<model_id>/<subdir>/<image>.png", ...]}
"""

import argparse
import json
from collections import OrderedDict, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group matched per-image result keys by model_id"
    )
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--include-empty", action="store_true")
    return parser.parse_args()


def extract_model_id_from_image_path(path: str) -> str:
    parts = [part for part in str(path).strip().split("/") if part]
    if len(parts) < 3:
        raise ValueError(f"cannot extract model_id from path: {path}")
    # Expect .../<model_id>/<subdir>/<filename>
    return parts[-3]


def main():
    args = parse_args()
    grouped = defaultdict(list)
    seen = defaultdict(set)
    invalid = 0
    total = 0

    with open(args.source_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            s = (line or "").strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except Exception:
                invalid += 1
                continue
            if not isinstance(obj, dict) or len(obj) != 1:
                invalid += 1
                continue
            image_path = next(iter(obj.keys()))
            if not isinstance(image_path, str) or not image_path.strip():
                invalid += 1
                continue
            image_path = image_path.strip()
            try:
                model_id = extract_model_id_from_image_path(image_path)
            except Exception:
                invalid += 1
                continue
            if image_path not in seen[model_id]:
                seen[model_id].add(image_path)
                grouped[model_id].append(image_path)

    ordered = OrderedDict((model_id, grouped[model_id]) for model_id in sorted(grouped))

    written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for model_id, paths in ordered.items():
            if not paths and not args.include_empty:
                continue
            fout.write(json.dumps({model_id: paths}, ensure_ascii=False) + "\n")
            written += 1

    print(f"source_jsonl={args.source_jsonl}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"total_rows={total}")
    print(f"invalid_rows={invalid}")
    print(f"model_ids={len(ordered)}")
    print(f"written={written}")
    print(f"total_paths={sum(len(v) for v in ordered.values())}")


if __name__ == "__main__":
    main()

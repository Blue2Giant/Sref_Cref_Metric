#!/usr/bin/env python3
"""
Intersect matched per-image jsonl keys from content/style results and group by model combo.

Input jsonl format:
{"<image_path>": [...]}

Output jsonl format:
{"<model_combo>": ["<image_path_a>", "<image_path_b>", ...]}
"""

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intersect content/style matched keys and regroup by model combo"
    )
    parser.add_argument("--content-jsonl", required=True)
    parser.add_argument("--style-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--include-empty", action="store_true")
    return parser.parse_args()


def read_key_set(path: str) -> set[str]:
    out = set()
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict) or len(obj) != 1:
                continue
            key = next(iter(obj.keys()))
            if isinstance(key, str) and key.strip():
                out.add(key.strip())
    return out


def extract_model_combo(image_path: str) -> str:
    parts = [part for part in str(image_path).strip().split("/") if part]
    if len(parts) < 3:
        raise ValueError(f"cannot extract model combo from path: {image_path}")
    return parts[-3]


def main():
    args = parse_args()
    content_jsonl = Path(args.content_jsonl)
    style_jsonl = Path(args.style_jsonl)
    out_jsonl = Path(args.out_jsonl)

    if not content_jsonl.is_file():
        raise RuntimeError(f"content jsonl not found: {content_jsonl}")
    if not style_jsonl.is_file():
        raise RuntimeError(f"style jsonl not found: {style_jsonl}")

    content_keys = read_key_set(str(content_jsonl))
    style_keys = read_key_set(str(style_jsonl))
    intersected_keys = sorted(content_keys & style_keys)

    grouped = defaultdict(list)
    invalid_paths = 0
    for image_path in intersected_keys:
        try:
            combo = extract_model_combo(image_path)
        except Exception:
            invalid_paths += 1
            continue
        grouped[combo].append(image_path)

    ordered = OrderedDict((combo, grouped[combo]) for combo in sorted(grouped))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for combo, paths in ordered.items():
            if not paths and not args.include_empty:
                continue
            fout.write(json.dumps({combo: paths}, ensure_ascii=False) + "\n")
            written += 1

    stats = {
        "content_jsonl": str(content_jsonl),
        "style_jsonl": str(style_jsonl),
        "out_jsonl": str(out_jsonl),
        "content_key_count": len(content_keys),
        "style_key_count": len(style_keys),
        "intersected_key_count": len(intersected_keys),
        "invalid_paths": invalid_paths,
        "model_combo_count": len(ordered),
        "written_lines": written,
        "avg_images_per_model_combo": (len(intersected_keys) / len(ordered)) if ordered else 0.0,
    }

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Filter dual-lora triplet jsonl rows by one-lora content/style model-id key sets.

Keep a dual key "a__b" if either:
1. a is in content ids and b is in style ids
2. a is in style ids and b is in content ids

The input jsonl is expected to be one JSON object per line with a single key.
Matched lines are written out verbatim so the original value payload is preserved.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter dual-lora triplet jsonl by one-lora content/style key sets"
    )
    parser.add_argument("--content-jsonl", required=True)
    parser.add_argument("--style-jsonl", required=True)
    parser.add_argument("--dual-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--stats-json", default="")
    return parser.parse_args()


def read_single_key_jsonl_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            raw = (line or "").strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as exc:
                raise RuntimeError(f"invalid json at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict) or len(obj) != 1:
                raise RuntimeError(f"expected single-key object at {path}:{line_no}")
            key = next(iter(obj.keys()))
            if not isinstance(key, str) or not key.strip():
                raise RuntimeError(f"expected non-empty string key at {path}:{line_no}")
            keys.add(key.strip())
    return keys


def parse_dual_key(key: str) -> tuple[str, str]:
    parts = key.split("__")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid dual key: {key}")
    return parts[0], parts[1]


def main():
    args = parse_args()

    content_jsonl = Path(args.content_jsonl)
    style_jsonl = Path(args.style_jsonl)
    dual_jsonl = Path(args.dual_jsonl)
    out_jsonl = Path(args.out_jsonl)

    for path in (content_jsonl, style_jsonl, dual_jsonl):
        if not path.is_file():
            raise RuntimeError(f"input jsonl not found: {path}")

    content_ids = read_single_key_jsonl_keys(content_jsonl)
    style_ids = read_single_key_jsonl_keys(style_jsonl)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    malformed_rows = 0
    left_content_right_style = 0
    left_style_right_content = 0

    with dual_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            raw = (line or "").strip()
            if not raw:
                continue
            total_rows += 1
            try:
                obj = json.loads(raw)
            except Exception:
                malformed_rows += 1
                continue
            if not isinstance(obj, dict) or len(obj) != 1:
                malformed_rows += 1
                continue

            key = next(iter(obj.keys()))
            if not isinstance(key, str):
                malformed_rows += 1
                continue

            try:
                left_id, right_id = parse_dual_key(key.strip())
            except ValueError:
                malformed_rows += 1
                continue

            keep_lc_rs = left_id in content_ids and right_id in style_ids
            keep_ls_rc = left_id in style_ids and right_id in content_ids
            if not (keep_lc_rs or keep_ls_rc):
                continue

            if keep_lc_rs:
                left_content_right_style += 1
            if keep_ls_rc:
                left_style_right_content += 1

            fout.write(line if line.endswith("\n") else f"{line}\n")
            kept_rows += 1

    stats = {
        "content_jsonl": str(content_jsonl),
        "style_jsonl": str(style_jsonl),
        "dual_jsonl": str(dual_jsonl),
        "out_jsonl": str(out_jsonl),
        "content_id_count": len(content_ids),
        "style_id_count": len(style_ids),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows": total_rows - kept_rows,
        "malformed_rows": malformed_rows,
        "left_content_right_style": left_content_right_style,
        "left_style_right_content": left_style_right_content,
    }

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

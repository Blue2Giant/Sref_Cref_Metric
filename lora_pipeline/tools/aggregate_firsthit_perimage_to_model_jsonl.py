#!/usr/bin/env python3
"""
Aggregate per-image first-hit judge outputs back to model-level jsonl.

Input prepared triplet jsonl:
{"model_id__model_id": ["/path/a.png", "/path/b.png", ...]}

Input judge outputs:
- matched jsonl: {"/path/a.png": [...]}
- optional all-similar jsonl: {"/path/b.png": [...]}
- processed jsonl: {"/path/a.png": {"bucket": "matched"|"all_similar"|"no_match"}}
- optional error log jsonl

Output:
- true jsonl: {"model_id": [true_image_paths...]}
- fail jsonl: {"model_id": [fail_image_paths...]}
- optional error jsonl: {"model_id": [error_or_unprocessed_image_paths...]}
"""

import argparse
import json
from collections import OrderedDict, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-image first-hit outputs into model-level true/fail jsonl"
    )
    parser.add_argument("--prepared-triplet-jsonl", required=True)
    parser.add_argument("--matched-jsonl", required=True)
    parser.add_argument("--processed-jsonl", default="")
    parser.add_argument("--all-similar-jsonl", default="")
    parser.add_argument("--error-log-jsonl", default="")
    parser.add_argument("--out-true-jsonl", required=True)
    parser.add_argument("--out-fail-jsonl", required=True)
    parser.add_argument("--out-error-jsonl", default="")
    parser.add_argument("--pair-sep", default="__")
    parser.add_argument("--include-empty", action="store_true")
    return parser.parse_args()


def read_jsonl_key_set(path: str) -> set[str]:
    out = set()
    if not path:
        return out
    from pathlib import Path
    if not Path(path).is_file():
        return out
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict) and len(obj) == 1:
                key = next(iter(obj))
                if isinstance(key, str) and key.strip():
                    out.add(key.strip())
    return out


def read_processed_bucket_map(path: str) -> dict[str, str]:
    out = {}
    if not path:
        return out
    from pathlib import Path
    if not Path(path).is_file():
        return out
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
            key, value = next(iter(obj.items()))
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            bucket = value.get("bucket")
            if isinstance(bucket, str) and bucket.strip():
                out[key.strip()] = bucket.strip()
    return out


def read_error_key_set(path: str) -> set[str]:
    out = set()
    if not path:
        return out
    from pathlib import Path
    if not Path(path).is_file():
        return out
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            result_key = obj.get("result_key", "")
            if isinstance(result_key, str) and result_key.strip():
                out.add(result_key.strip())
    return out


def read_prepared_triplet_map(path: str, pair_sep: str) -> OrderedDict[str, list[str]]:
    out = OrderedDict()
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = (line or "").strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict) or len(obj) != 1:
                continue
            pair_key, value = next(iter(obj.items()))
            if not isinstance(pair_key, str) or not isinstance(value, list):
                continue
            pair_key = pair_key.strip()
            if not pair_key:
                continue
            if pair_sep in pair_key:
                model_id = pair_key.split(pair_sep, 1)[0].strip()
            else:
                model_id = pair_key
            if not model_id:
                continue
            paths = []
            seen = set()
            for path_item in value:
                if not isinstance(path_item, str):
                    continue
                path_item = path_item.strip()
                if not path_item or path_item in seen:
                    continue
                seen.add(path_item)
                paths.append(path_item)
            out[model_id] = paths
    return out


def write_grouped_jsonl(path: str, grouped: OrderedDict[str, list[str]], include_empty: bool):
    written = 0
    with open(path, "w", encoding="utf-8") as fout:
        for model_id, paths in grouped.items():
            if not paths and not include_empty:
                continue
            fout.write(json.dumps({model_id: paths}, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    args = parse_args()

    model_to_paths = read_prepared_triplet_map(args.prepared_triplet_jsonl, pair_sep=args.pair_sep)
    matched_keys = read_jsonl_key_set(args.matched_jsonl)
    all_similar_keys = read_jsonl_key_set(args.all_similar_jsonl)
    processed_buckets = read_processed_bucket_map(args.processed_jsonl)
    error_keys = read_error_key_set(args.error_log_jsonl)

    true_grouped = OrderedDict()
    fail_grouped = OrderedDict()
    error_grouped = OrderedDict()

    total_images = 0
    true_images = 0
    fail_images = 0
    error_images = 0

    for model_id, paths in model_to_paths.items():
        true_paths = []
        fail_paths = []
        err_paths = []
        for path in paths:
            total_images += 1
            bucket = processed_buckets.get(path, "")
            is_true = path in matched_keys or path in all_similar_keys or bucket in {"matched", "all_similar"}
            is_fail = bucket == "no_match"
            is_error = (path in error_keys) or (not is_true and not is_fail and bucket != "")

            if is_true:
                true_paths.append(path)
                true_images += 1
            elif is_fail:
                fail_paths.append(path)
                fail_images += 1
            else:
                err_paths.append(path)
                error_images += 1

        true_grouped[model_id] = true_paths
        fail_grouped[model_id] = fail_paths
        error_grouped[model_id] = err_paths

    true_written = write_grouped_jsonl(args.out_true_jsonl, true_grouped, include_empty=args.include_empty)
    fail_written = write_grouped_jsonl(args.out_fail_jsonl, fail_grouped, include_empty=args.include_empty)
    error_written = 0
    if args.out_error_jsonl:
        error_written = write_grouped_jsonl(args.out_error_jsonl, error_grouped, include_empty=args.include_empty)

    print(f"prepared_triplet_jsonl={args.prepared_triplet_jsonl}")
    print(f"matched_jsonl={args.matched_jsonl}")
    if args.processed_jsonl:
        print(f"processed_jsonl={args.processed_jsonl}")
    if args.all_similar_jsonl:
        print(f"all_similar_jsonl={args.all_similar_jsonl}")
    if args.error_log_jsonl:
        print(f"error_log_jsonl={args.error_log_jsonl}")
    print(f"out_true_jsonl={args.out_true_jsonl}")
    print(f"out_fail_jsonl={args.out_fail_jsonl}")
    if args.out_error_jsonl:
        print(f"out_error_jsonl={args.out_error_jsonl}")
    print(f"model_ids={len(model_to_paths)}")
    print(f"total_images={total_images}")
    print(f"true_images={true_images}")
    print(f"fail_images={fail_images}")
    print(f"error_images={error_images}")
    print(f"true_written={true_written}")
    print(f"fail_written={fail_written}")
    if args.out_error_jsonl:
        print(f"error_written={error_written}")


if __name__ == "__main__":
    main()

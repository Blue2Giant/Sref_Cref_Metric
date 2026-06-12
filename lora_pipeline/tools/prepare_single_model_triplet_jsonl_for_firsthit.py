#!/usr/bin/env python3
"""
Prepare a single-model-id image jsonl for first-hit judge scripts.

Input format:
{"<model_id>": ["/abs/path/a.png", "s3://.../b.png", ...]}

Output format:
{"<model_id>__<model_id>": ["/abs/path/a.png", "s3://.../b.png", ...]}

Only model_id values present in the whitelist txt are kept.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a single-model triplet jsonl by model-id whitelist and rewrite keys as self-pairs"
    )
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--model-id-txt", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--missing-model-id-txt", default="")
    parser.add_argument("--pair-sep", default="__")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def read_model_ids(path: str) -> tuple[list[str], set[str]]:
    ordered = []
    seen = set()
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            model_id = line.strip()
            if not model_id or model_id.startswith("#") or model_id in seen:
                continue
            ordered.append(model_id)
            seen.add(model_id)
    return ordered, seen


def main():
    args = parse_args()
    source_jsonl = Path(args.source_jsonl)
    out_jsonl = Path(args.out_jsonl)
    missing_txt = Path(args.missing_model_id_txt) if args.missing_model_id_txt else None

    if not source_jsonl.is_file():
        raise RuntimeError(f"source jsonl does not exist: {source_jsonl}")

    ordered_model_ids, allowed_model_ids = read_model_ids(args.model_id_txt)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if missing_txt is not None:
        missing_txt.parent.mkdir(parents=True, exist_ok=True)

    emitted_model_ids = set()
    duplicate_hits = 0
    invalid_lines = 0
    written = 0
    total_lines = 0

    with source_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            total_lines += 1
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                invalid_lines += 1
                continue
            if not isinstance(obj, dict) or len(obj) != 1:
                invalid_lines += 1
                continue

            model_id, paths = next(iter(obj.items()))
            if not isinstance(model_id, str) or not isinstance(paths, list):
                invalid_lines += 1
                continue
            model_id = model_id.strip()
            if not model_id:
                invalid_lines += 1
                continue
            if args.pair_sep in model_id:
                raise RuntimeError(
                    f"source jsonl contains pair-style key at line {line_no}: {model_id}. "
                    "This tool expects single model_id keys."
                )
            if model_id not in allowed_model_ids:
                continue
            if model_id in emitted_model_ids:
                duplicate_hits += 1
                continue

            filtered_paths = [str(path).strip() for path in paths if isinstance(path, str) and str(path).strip()]
            if not filtered_paths:
                continue

            pair_key = f"{model_id}{args.pair_sep}{model_id}"
            fout.write(json.dumps({pair_key: filtered_paths}, ensure_ascii=False) + "\n")
            emitted_model_ids.add(model_id)
            written += 1

            if args.progress_every > 0 and written % args.progress_every == 0:
                print(
                    f"progress written={written} allowed={len(allowed_model_ids)} duplicates={duplicate_hits} invalid={invalid_lines}",
                    flush=True,
                )

    missing_model_ids = [model_id for model_id in ordered_model_ids if model_id not in emitted_model_ids]
    if missing_txt is not None:
        missing_txt.write_text(
            "\n".join(missing_model_ids) + ("\n" if missing_model_ids else ""),
            encoding="utf-8",
        )

    print(f"source_jsonl={source_jsonl}")
    print(f"model_id_txt={args.model_id_txt}")
    print(f"out_jsonl={out_jsonl}")
    print(f"allowed_model_ids={len(allowed_model_ids)}")
    print(f"written={written}")
    print(f"missing={len(missing_model_ids)}")
    print(f"duplicate_hits={duplicate_hits}")
    print(f"invalid_lines={invalid_lines}")
    print(f"total_lines={total_lines}")
    if missing_txt is not None:
        print(f"missing_model_id_txt={missing_txt}")


if __name__ == "__main__":
    main()

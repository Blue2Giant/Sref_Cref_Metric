#!/usr/bin/env python3
"""
找到两个jsonl判别都为true的交集：
/data/benchmark_metrics/logs/triplet_content_style_intersection_non_empty_keys.txt
"""
import argparse
import json
from pathlib import Path
from typing import Set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-a", required=True)
    p.add_argument("--jsonl-b", required=True)
    p.add_argument("--out-txt", required=True)
    return p.parse_args()


def load_non_empty_keys(path: Path) -> Set[str]:
    out: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict) or len(obj) != 1:
                continue
            k, v = next(iter(obj.items()))
            if isinstance(k, str) and isinstance(v, list) and len(v) > 0:
                out.add(k)
    return out


def main():
    args = parse_args()
    a_path = Path(args.jsonl_a)
    b_path = Path(args.jsonl_b)
    out_path = Path(args.out_txt)
    if not a_path.is_file():
        raise RuntimeError(f"jsonl-a 不存在: {a_path}")
    if not b_path.is_file():
        raise RuntimeError(f"jsonl-b 不存在: {b_path}")

    a_keys = load_non_empty_keys(a_path)
    b_keys = load_non_empty_keys(b_path)
    inter = sorted(a_keys & b_keys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(inter) + ("\n" if inter else ""), encoding="utf-8")

    print(f"a_non_empty={len(a_keys)}")
    print(f"b_non_empty={len(b_keys)}")
    print(f"intersection={len(inter)}")
    print(f"out_txt={out_path}")


if __name__ == "__main__":
    main()

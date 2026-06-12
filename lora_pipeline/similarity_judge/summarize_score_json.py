#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description="统计 content/style 打分 JSON")
    ap.add_argument("--score_json", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    with open(args.score_json, "r", encoding="utf-8") as f:
        scores = json.load(f)

    values = [v for v in scores.values() if isinstance(v, int)]
    total = len(scores)
    valid = len(values)
    null_count = sum(v is None for v in scores.values())
    avg = (sum(values) / valid) if valid > 0 else None
    dist = dict(sorted(Counter(values).items()))
    top_items = sorted(
        ((k, v) for k, v in scores.items() if isinstance(v, int)),
        key=lambda x: (-x[1], x[0]),
    )[: args.top_k]

    print(f"total={total}")
    print(f"valid={valid}")
    print(f"null={null_count}")
    print(f"avg={avg:.6f}" if avg is not None else "avg=None")
    print(f"dist={json.dumps(dist, ensure_ascii=False)}")
    print("top_items=")
    for key, score in top_items:
        print(f"{score}\t{key}")


if __name__ == "__main__":
    main()

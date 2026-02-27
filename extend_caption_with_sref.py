#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/benchmark_metrics/extend_caption_with_sref.py /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new/prompts_dual_en.json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new/prompts.json --seed 42
"""
import json
import random
import argparse
from pathlib import Path

STYLE_SENTENCES = [
    "Transfer the style into the style reference picture.",
    "Transfer the style into the style reference image.",
    "Transfer the style to the style reference picture.",
    "Apply the style to the style reference picture.",
    "Adopt the style from the style reference picture.",
    "Embrace the aesthetic of the style reference picture.",
    "Incorporate the style from the style reference image.",
    "Reflect the style of the style reference picture.",
    "Capture the essence of the style reference image.",
    "Utilize the style from the style reference picture.",
]

def process_json(input_path: str, output_path: str) -> None:
    # 读入原始 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}

    for k, v in data.items():
        # 随机选一句英文
        suffix = random.choice(STYLE_SENTENCES)

        # 如果原来的 value 已经是字符串，直接拼接
        if isinstance(v, str):
            new_value = v.strip()
            # 中英文之间加一个空格，避免黏在一起
            new_value = f"{new_value} {suffix}"
        else:
            # 如果不是字符串，先转成字符串再加
            new_value = f"{str(v).strip()} {suffix}"

        new_data[k] = new_value

    # 写出新的 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Append a random style sentence to each value in a JSON file."
    )
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (optional)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    process_json(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
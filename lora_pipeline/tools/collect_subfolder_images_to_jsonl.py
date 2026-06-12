#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把某个目录下每个直接子文件夹中的图片收集成 JSONL。

输出格式：
{"子文件夹名": ["/abs/path/a.png", "/abs/path/b.jpg", ...]}

示例：
python3 /data/benchmark_metrics/lora_pipeline/tools/collect_subfolder_images_to_jsonl.py \
  --root-dir /mnt/jfs/loras_combine/qwen_0323_dual_lora \
  --out-jsonl /data/benchmark_metrics/logs/triplet_jsonl/qwen_0323_dual_lora_images_by_subfolder.jsonl \
  --recursive
"""
import argparse
import json
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def parse_args():
    p = argparse.ArgumentParser(description="收集每个直接子文件夹中的图片绝对路径，输出为 jsonl")
    p.add_argument("--root-dir", required=True, help="根目录；key 使用它的直接子文件夹名称")
    p.add_argument("--out-jsonl", required=True, help="输出 jsonl 路径")
    p.add_argument("--ext", action="append", default=[], help="额外指定图片扩展名，可重复传入")
    p.add_argument("--recursive", action="store_true", help="递归搜索每个子文件夹下的图片")
    p.add_argument("--include-empty", action="store_true", help="没有图片的子文件夹也写出空列表")
    p.add_argument("--progress-every", type=int, default=200, help="每处理多少个子文件夹打印一次进度，0 表示不打印")
    return p.parse_args()


def normalize_exts(exts: Iterable[str]) -> set[str]:
    out = set(DEFAULT_EXTS)
    for ext in exts:
        s = str(ext).strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        out.add(s)
    return out


def is_image(path: Path, exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def list_images(folder: Path, recursive: bool, exts: set[str]) -> List[str]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    images = [str(p.resolve()) for p in iterator if is_image(p, exts)]
    images.sort()
    return images


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    out_jsonl = Path(args.out_jsonl)
    exts = normalize_exts(args.ext)

    if not root_dir.is_dir():
        raise RuntimeError(f"root-dir 不存在: {root_dir}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    total_subdirs = len(subdirs)
    written = 0
    empty = 0

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for idx, subdir in enumerate(subdirs, start=1):
            images = list_images(subdir, recursive=bool(args.recursive), exts=exts)
            if not images:
                empty += 1
                if not args.include_empty:
                    if args.progress_every > 0 and idx % int(args.progress_every) == 0:
                        print(f"progress {idx}/{total_subdirs} written={written} empty={empty}", flush=True)
                    continue
            fout.write(json.dumps({subdir.name: images}, ensure_ascii=False) + "\n")
            written += 1
            if args.progress_every > 0 and idx % int(args.progress_every) == 0:
                print(f"progress {idx}/{total_subdirs} written={written} empty={empty}", flush=True)

    print(f"root_dir={root_dir}")
    print(f"out_jsonl={out_jsonl}")
    print(f"subdirs_total={total_subdirs}")
    print(f"written={written}")
    print(f"empty={empty}")


if __name__ == "__main__":
    main()

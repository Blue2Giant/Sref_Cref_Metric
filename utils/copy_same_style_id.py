#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 SRC_ROOT 下形如 style_id__content_id 的目录重组为：

SRC_ROOT/style_id__content_id/
  style_100/
  content_100/
  two_100/

→

DST_ROOT/style_id/
  style_100/        # 拷一次（任意一个 pair 的 style_100）
  <content_id>/     # 每个 content_id 一个目录
    ...             # 来自原来 two_100/ 的所有文件

用 megfile 支持 s3:// / s3+b:// / 本地路径。
python /data/LoraPipeline/utils/copy_same_style_id.py  --src-root s3://lanjinghong-data/loras_eval_qwen_two_lora_with_trigger_merge  --dst-root s3://lanjinghong-data/loras_eval_qwen_two_lora_group_by_style \
  --overwrite

"""

import os
import re
import argparse
from typing import List, Set

from megfile.smart import (
    smart_listdir,
    smart_exists,
    smart_isdir,
    smart_makedirs,
    smart_copy as mcopy,
)
from tqdm import tqdm


def join_path(root: str, name: str) -> str:
    return root.rstrip("/") + "/" + name.lstrip("/")


def copy_all_files(src_dir: str, dst_dir: str, overwrite: bool = False) -> int:
    """
    把 src_dir 下的所有“文件”拷贝到 dst_dir（不递归子目录）。
    返回拷贝的文件数量。
    """
    if not smart_exists(src_dir):
        return 0

    smart_makedirs(dst_dir, exist_ok=True)
    count = 0

    try:
        names = smart_listdir(src_dir)
    except FileNotFoundError:
        return 0

    for name in names:
        src = join_path(src_dir, name)
        # 跳过子目录，只拷文件
        if smart_isdir(src):
            continue
        dst = join_path(dst_dir, name)
        try:
            mcopy(src, dst, overwrite=overwrite)
            count += 1
        except Exception as e:
            print(f"[WARN] 拷贝失败: {src} -> {dst} ({e})")

    return count


def regroup(src_root: str, dst_root: str, overwrite: bool = False):
    if not smart_exists(src_root):
        print(f"[ERROR] src_root 不存在: {src_root}")
        return

    smart_makedirs(dst_root, exist_ok=True)

    # 记录哪些 style_id 的 style_100 已经拷过，避免重复
    styled_done: Set[str] = set()

    try:
        names = smart_listdir(src_root)
    except FileNotFoundError:
        print(f"[ERROR] 无法列出目录: {src_root}")
        return

    # 只看形如 "<style_id>__<content_id>" 的目录
    pair_dirs: List[str] = []
    for name in names:
        m = re.match(r"^(\d+)__(\d+)$", name)
        if not m:
            continue
        full = join_path(src_root, name)
        if smart_isdir(full):
            pair_dirs.append(name)

    if not pair_dirs:
        print(f"[INFO] 在 {src_root} 下没有找到 style_id__content_id 结构，直接退出。")
        return

    print(f"[INFO] 在 {src_root} 下找到 {len(pair_dirs)} 个 style_id__content_id 目录")

    for name in tqdm(pair_dirs, desc="Re-group pairs"):
        m = re.match(r"^(\d+)__(\d+)$", name)
        style_id, content_id = m.group(1), m.group(2)

        src_pair_dir = join_path(src_root, name)
        src_two_dir = join_path(src_pair_dir, "two_100")
        src_style_dir = join_path(src_pair_dir, "style_100")

        # 目标路径
        dst_style_root = join_path(dst_root, style_id)
        dst_content_dir = join_path(dst_style_root, content_id)
        dst_style_dir = join_path(dst_style_root, "style_100")

        # two_100 → style_id/content_id/
        if smart_exists(src_two_dir):
            copied = copy_all_files(src_two_dir, dst_content_dir, overwrite=overwrite)
            print(
                f"[INFO] {name}: two_100 -> {style_id}/{content_id}/, "
                f"拷贝 {copied} 个文件"
            )
        else:
            print(f"[WARN] {name}: 找不到 two_100/，跳过 two_100")

        # style_100 → style_id/style_100/（每个 style_id 只拷一次）
        if smart_exists(src_style_dir):
            if style_id not in styled_done or overwrite:
                copied = copy_all_files(src_style_dir, dst_style_dir, overwrite=overwrite)
                styled_done.add(style_id)
                print(
                    f"[INFO] {name}: style_100 -> {style_id}/style_100/, "
                    f"拷贝 {copied} 个文件"
                )
        else:
            print(f"[WARN] {name}: 找不到 style_100/，跳过 style_100")


def main():
    parser = argparse.ArgumentParser(
        description="按照 style_id 归并 style_id__content_id 目录下的 two_100/style_100"
    )
    parser.add_argument(
        "--src-root",
        required=True,
        help="原始根目录（例如 s3://lanjinghong-data/loras_eval_qwen_two_lora_with_trigger_merge）",
    )
    parser.add_argument(
        "--dst-root",
        required=True,
        help="输出根目录（可以是另一个桶路径，也可以是本地目录）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如指定，则允许覆盖 dst 下已有同名文件",
    )
    args = parser.parse_args()

    regroup(args.src_root.rstrip("/"), args.dst_root.rstrip("/"), overwrite=args.overwrite)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse flat selection JSONL generated from copy_one_lora_2see_flat.py back to source paths.

Example:
python /data/benchmark_metrics/lora_pipeline/tools/reverse_flat_selections_to_source_jsonl.py \
  --selection-jsonl /data/benchmark_metrics/logs/triplet_jsonl/selections.jsonl \
  --eval-root s3://lanjinghong-data/loras_eval_illustrious_one_img_magic \
  --one-lora-root /mnt/jfs/loras_combine/illustrious_0321_two_lora \
  --eval-subfolder eval_images_with_negative_new \
  --one-subfolder eval_images_with_negative_new \
  --out-jsonl /data/benchmark_metrics/logs/triplet_jsonl/selections_original_paths.jsonl
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

try:
    import megfile as mf
except Exception as e:
    raise RuntimeError(
        "需要 megfile 才能同时处理 s3:// 与本地路径。请先安装：pip install megfile\n"
        f"import megfile failed: {e}"
    )


IMAGE_EXTS_DEFAULT = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def smart_exists(path: str) -> bool:
    return mf.smart_exists(path)


def smart_isdir(path: str) -> bool:
    return mf.smart_isdir(path)


def smart_join(root: str, name: str) -> str:
    if root.endswith("/"):
        return root + name
    return root + "/" + name


def smart_listdir(path: str) -> List[str]:
    items = mf.smart_listdir(path)
    out: List[str] = []
    for x in items:
        x = str(x).rstrip("/")
        out.append(os.path.basename(x))
    return out


def is_image(name: str, exts: List[str]) -> bool:
    low = name.lower()
    return any(low.endswith(ext) for ext in exts)


def list_images(dir_path: str, exts: List[str]) -> List[str]:
    if (not smart_exists(dir_path)) or (not smart_isdir(dir_path)):
        return []
    names = smart_listdir(dir_path)
    imgs = [n for n in names if is_image(n, exts)]
    imgs.sort()
    return imgs


def parse_flat_path(flat_path: str) -> Tuple[str, int]:
    parent = os.path.basename(os.path.dirname(flat_path.rstrip("/")))
    m = re.fullmatch(r"(eval|one_lora)_(\d+)", parent)
    if not m:
        raise ValueError(f"无法从路径识别 eval/one_lora 索引: {flat_path}")
    kind = m.group(1)
    idx = int(m.group(2))
    if idx <= 0:
        raise ValueError(f"索引必须从 1 开始: {flat_path}")
    return kind, idx - 1


def read_jsonl(path: str) -> List[Dict[str, List[str]]]:
    rows: List[Dict[str, List[str]]] = []
    with mf.smart_open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = (line or "").strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception as e:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {e}") from e
            if not isinstance(row, dict) or len(row) != 1:
                raise ValueError(f"第 {line_no} 行格式不合法，期望单 key 字典: {row}")
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="把扁平选择图片路径 jsonl 反推成原始源路径 jsonl，并校验存在性。")
    ap.add_argument("--selection-jsonl", required=True, help="输入 selections.jsonl")
    ap.add_argument("--eval-root", required=True, help="eval 侧源根目录")
    ap.add_argument("--one-lora-root", required=True, help="one_lora 侧源根目录")
    ap.add_argument("--eval-subfolder", required=True, help="eval 侧模型子目录名")
    ap.add_argument("--one-subfolder", required=True, help="one_lora 侧模型子目录名")
    ap.add_argument("--out-jsonl", required=True, help="输出源路径 jsonl")
    ap.add_argument("--exts", default=",".join(IMAGE_EXTS_DEFAULT), help="图片后缀，逗号分隔")
    args = ap.parse_args()

    exts = [x.strip().lower() for x in args.exts.split(",") if x.strip()]
    rows = read_jsonl(args.selection_jsonl)
    cache: Dict[Tuple[str, str], List[str]] = {}
    errors: List[str] = []
    total_paths = 0

    with mf.smart_open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for row_idx, row in enumerate(rows, start=1):
            model_id, flat_paths = next(iter(row.items()))
            if not isinstance(flat_paths, list):
                raise ValueError(f"第 {row_idx} 行 model_id={model_id} 的 value 不是 list")

            source_paths: List[str] = []
            for flat_path in flat_paths:
                kind, zero_based_idx = parse_flat_path(flat_path)
                if kind == "eval":
                    src_dir = smart_join(
                        smart_join(args.eval_root.rstrip("/"), model_id),
                        args.eval_subfolder.strip("/"),
                    )
                else:
                    src_dir = smart_join(
                        smart_join(args.one_lora_root.rstrip("/"), model_id),
                        args.one_subfolder.strip("/"),
                    )

                cache_key = (kind, model_id)
                if cache_key not in cache:
                    imgs = list_images(src_dir, exts)
                    cache[cache_key] = imgs
                imgs = cache[cache_key]

                if zero_based_idx >= len(imgs):
                    errors.append(
                        f"model_id={model_id} flat_path={flat_path} 在源目录 {src_dir} 中索引越界: "
                        f"idx={zero_based_idx} total={len(imgs)}"
                    )
                    continue

                src_path = smart_join(src_dir, imgs[zero_based_idx])
                if not smart_exists(src_path):
                    errors.append(f"model_id={model_id} flat_path={flat_path} 源路径不存在: {src_path}")
                    continue

                source_paths.append(src_path)
                total_paths += 1

            out_f.write(json.dumps({model_id: source_paths}, ensure_ascii=False) + "\n")

    print(f"[INFO] 输入行数: {len(rows)}")
    print(f"[INFO] 成功反推路径数: {total_paths}")
    print(f"[INFO] 输出文件: {args.out_jsonl}")
    if errors:
        print(f"[ERROR] 发现 {len(errors)} 个问题:")
        for err in errors[:50]:
            print(err)
        if len(errors) > 50:
            print(f"... 其余 {len(errors) - 50} 个错误已省略")
        return 2

    print("[DONE] 所有反推源路径均已校验存在")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

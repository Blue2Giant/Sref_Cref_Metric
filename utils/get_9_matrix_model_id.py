#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 content_similarity / style_similarity 分布并导出排序 txt 和统计结果，
并且在前若干百分比范围内，为每个 model_id 从 content_100/style_100 中
采样 9 张图片，拼成 3x3 九宫格，保存为 model_id.png。

示例：
  python histogram_of_similarity.py \
    --root s3://lanjinghong-data/loras_eval_qwen \
    --out-dir /data/LoraPipeline/similarity_stats \
    --content-top-pct 30 \
    --style-top-pct 30 \
    --grid-num 9
"""

import os
import io
import json
import argparse
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from megfile.smart import (
    smart_listdir,
    smart_isdir,
    smart_exists,
    smart_open as mopen,
    smart_makedirs,
)

plt.switch_backend("Agg")  # 允许在无显示环境下画图（当前只用到 Image，不画直方图）

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_values(root: str) -> Tuple[List[float], List[float], Dict[str, float], Dict[str, float]]:
    """
    遍历 root/model_id/ 下的 json，收集：
      - content_similarity.json 里的 mean_similarity / overall_mean_similarity（仅 content==1）
      - style_similarity.json   里的 weighted_score           （仅 style==1）

    返回：
      content_vals: 所有内容相似度数值列表
      style_vals  : 所有画风相似度数值列表
      content_map : {model_id: content_similarity}
      style_map   : {model_id: style_similarity}
    """
    content_vals: List[float] = []
    style_vals: List[float] = []
    content_map: Dict[str, float] = {}
    style_map: Dict[str, float] = {}

    root = root.rstrip("/")

    for name in smart_listdir(root):
        model_dir = f"{root}/{name}"
        if not smart_isdir(model_dir):
            continue

        # 先读取 meta：<model_id>_img1.json
        meta_path = f"{model_dir}/{name}_img1.json"
        if not smart_exists(meta_path):
            print(f"[WARN] {name}: 缺少 {name}_img1.json，无法判断 content/style 标记，跳过该模型")
            continue

        try:
            with mopen(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[WARN] {name}: 读取 meta 失败: {e}")
            continue

        # 解析 style / content 标记（兼容 int/bool/字符串）
        try:
            is_style = int(meta.get("style", 0) or 0) == 1
        except Exception:
            is_style = bool(meta.get("style"))

        try:
            is_content = int(meta.get("content", 0) or 0) == 1
        except Exception:
            is_content = bool(meta.get("content"))

        # -------- content similarity（只看 content==1 的）---------
        if is_content:
            c_path = f"{model_dir}/content_similarity.json"
            if smart_exists(c_path):
                try:
                    with mopen(c_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 兼容 mean_similarity / overall_mean_similarity
                    v = data.get("mean_similarity", None)
                    if v is None:
                        v = data.get("overall_mean_similarity", None)

                    if isinstance(v, (int, float)):
                        val = float(v)
                        content_vals.append(val)
                        content_map[name] = val
                    else:
                        print(f"[WARN] {name}: content_similarity.json 中没有数值字段 mean_similarity/overall_mean_similarity")
                except Exception as e:
                    print(f"[WARN] 读取 {c_path} 失败: {e}")
            else:
                print(f"[WARN] {name}: content==1 但缺少 content_similarity.json")

        # -------- style similarity（只看 style==1 的）---------
        if is_style:
            s_path = f"{model_dir}/style_similarity.json"
            if smart_exists(s_path):
                try:
                    with mopen(s_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    v = data.get("weighted_score", None)
                    if isinstance(v, (int, float)):
                        val = float(v)
                        style_vals.append(val)
                        style_map[name] = val
                    else:
                        print(f"[WARN] {name}: style_similarity.json 中没有数值字段 weighted_score")
                except Exception as e:
                    print(f"[WARN] 读取 {s_path} 失败: {e}")
            else:
                print(f"[WARN] {name}: style==1 但缺少 style_similarity.json")

    return content_vals, style_vals, content_map, style_map


def compute_stats(values: List[float], name: str) -> dict:
    """计算均值、方差、若干分位数，以及建议阈值。"""
    if not values:
        print(f"[WARN] {name}: 没有数据")
        return {}

    arr = np.array(values, dtype=float)
    stats = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),  # median
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }

    stats["threshold_recommend"] = {
        "loose": stats["p25"],
        "medium": stats["p50"],
        "strict": stats["p75"],
    }

    print(f"\n===== {name} 统计结果 =====")
    for k, v in stats.items():
        if k == "threshold_recommend":
            continue
        print(f"{k:>4}: {v:.6f}" if isinstance(v, float) else f"{k:>4}: {v}")

    tr = stats["threshold_recommend"]
    print("建议阈值（可按需要选用）：")
    print(f"  宽松   loose  ≈ {tr['loose']:.6f}")
    print(f"  中等   medium ≈ {tr['medium']:.6f}")
    print(f"  严格   strict ≈ {tr['strict']:.6f}")

    return stats


def select_models_by_percent_range(
    score_map: Dict[str, float],
    percent_range: Tuple[float, float],
    descending: bool = True,
) -> List[str]:
    """
    根据分数排名百分比区间选择模型。

    score_map: {model_id: score}
    percent_range: (start_pct, end_pct)，区间在 [0,100]。
        - 按从高到低排序：
          0 表示最高分，100 表示最低分。
          例如 0,25 表示从高到低前 25% 的模型。
    """
    if not score_map:
        return []

    items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=descending)
    n = len(items)
    if n == 0:
        return []

    start_pct, end_pct = percent_range

    # 规范到 [0, 100]
    start_pct = max(0.0, min(100.0, start_pct))
    end_pct = max(0.0, min(100.0, end_pct))
    if start_pct > end_pct:
        start_pct, end_pct = end_pct, start_pct
    if start_pct == end_pct:
        end_pct = min(100.0, start_pct + 1.0)

    # 将百分比转换成索引区间
    start_idx = int(start_pct / 100.0 * n)
    end_idx = int(end_pct / 100.0 * n)
    if end_idx <= start_idx:
        end_idx = min(n, start_idx + 1)

    start_idx = max(0, min(n, start_idx))
    end_idx = max(0, min(n, end_idx))

    sub_items = items[start_idx:end_idx]
    selected_ids = [mid for mid, _ in sub_items]

    print(
        f"[INFO] 根据排名百分比区间 {start_pct:.1f}–{end_pct:.1f} 选出 {len(selected_ids)}/{n} 个模型"
    )

    return selected_ids


def pick_image_paths_for_model(img_dir: str, num_images: int) -> List[str]:
    """
    从 img_dir 中选出 num_images 张图片（按文件名排序后做均匀抽样）。
    支持远程桶 + 本地路径。
    """
    if not smart_isdir(img_dir):
        return []

    try:
        names = smart_listdir(img_dir)
    except Exception as e:
        print(f"[WARN] 无法列出目录 {img_dir}: {e}")
        return []

    files = [
        n for n in names
        if os.path.splitext(n)[1].lower() in IMG_EXTS
    ]
    if not files:
        return []

    files.sort()
    if len(files) <= num_images:
        chosen = files
    else:
        # 均匀抽样
        step = len(files) / float(num_images)
        idxs = []
        for i in range(num_images):
            idx = int(i * step)
            if idx >= len(files):
                idx = len(files) - 1
            idxs.append(idx)
        idxs = sorted(set(idxs))
        chosen = [files[i] for i in idxs][:num_images]

    full_paths = [f"{img_dir.rstrip('/')}/{fn}" for fn in chosen]
    return full_paths


def build_single_model_grid(
    root: str,
    model_id: str,
    mode: str,
    num_images: int,
    out_path: str,
    tile_size: Tuple[int, int] = (256, 256),
):
    """
    为单个 model_id 构建一张 3x3 九宫格图：
      - mode="content" 时从 root/model_id/content_100 下取图
      - mode="style"   时从 root/model_id/style_100 下取图
      - 最多采样 num_images（一般设为 9）
      - 保存为 out_path（例如 out_dir/top_content_grids/<model_id>.png）
    """
    subdir = f"{root.rstrip('/')}/{model_id}"
    if mode == "content":
        img_dir = f"{subdir}/content_100"
    else:
        img_dir = f"{subdir}/style_100"

    img_paths = pick_image_paths_for_model(img_dir, num_images)
    if not img_paths:
        print(f"[WARN] {model_id}: 在 {img_dir} 下未找到图片，跳过九宫格")
        return

    imgs: List[Image.Image] = []
    for p in img_paths:
        try:
            with mopen(p, "rb") as f:
                data = f.read()
            im = Image.open(io.BytesIO(data))
            im = im.convert("RGB")
            im = ImageOps.fit(im, tile_size, Image.Resampling.LANCZOS)
            imgs.append(im)
        except Exception as e:
            print(f"[WARN] 读取图片失败 {p}: {e}")

    if not imgs:
        print(f"[WARN] {model_id}: 找到图片路径但加载都失败，跳过九宫格")
        return

    # 固定 3x3 九宫格，多余的忽略，不足的用白底空位
    rows = cols = 3
    w, h = tile_size
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))

    for idx, im in enumerate(imgs[: rows * cols]):
        r = idx // cols
        c = idx % cols
        grid.paste(im, (c * w, r * h))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"[INFO] {mode} 九宫格已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="远程/本地根目录，例如 s3://lanjinghong-data/loras_eval_qwen",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="本地输出目录，用来保存统计 JSON、排序 txt 和九宫格图片",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="（保留参数，但当前不再画直方图）",
    )
    parser.add_argument(
        "--grid-num",
        type=int,
        default=9,
        help="每个 model_id 抽取的图片数量，默认 9（九宫格）",
    )
    parser.add_argument(
        "--content-top-pct",
        type=float,
        default=None,
        help="content 模式：从高到低前多少百分比的模型画九宫格，例如 10 表示前 10%%；为空则不画。",
    )
    parser.add_argument(
        "--style-top-pct",
        type=float,
        default=None,
        help="style 模式：从高到低前多少百分比的模型画九宫格，例如 5 表示前 5%%；为空则不画。",
    )
    parser.add_argument(
        "--model-id-txt",
        default=None,
        help="可选：只为该 txt 中列出的 model_id 生成九宫格（一行一个，可写纯数字或以数字开头的行）",
    )

    args = parser.parse_args()
    root = args.root.rstrip("/")
    out_dir = args.out_dir
    smart_makedirs(out_dir, exist_ok=True)

    print(f"[INFO] root    = {root}")
    print(f"[INFO] out-dir = {out_dir}")

    model_id_whitelist: Optional[set[str]] = None
    if args.model_id_txt:
        model_id_whitelist = set()
        try:
            with mopen(args.model_id_txt, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    m = re.match(r"(\d+)", s)
                    if m:
                        model_id_whitelist.add(m.group(1))
                    else:
                        model_id_whitelist.add(s)
        except Exception as e:
            print(f"[ERROR] 读取 model-id txt 失败: {args.model_id_txt} ({e})")
            return

        print(f"[INFO] 从 {args.model_id_txt} 读取到 {len(model_id_whitelist)} 个 model_id 白名单")

    content_vals, style_vals, content_map, style_map = collect_values(root)
    print(f"[INFO] 收集到 content 相似度数量（content==1）：{len(content_vals)}")
    print(f"[INFO] 收集到 style   相似度数量（style==1）  ：{len(style_vals)}")

    # 不画直方图，直接统计
    stats = {
        "content_similarity": compute_stats(content_vals, "content_similarity"),
        "style_weighted_score": compute_stats(style_vals, "style_weighted_score"),
    }

    stats_path = os.path.join(out_dir, "similarity_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 统计结果已写入: {stats_path}")

    # ========== 输出排序 txt ==========
    if content_map:
        content_rank_path = os.path.join(out_dir, "content_similarity_rank.txt")
        sorted_content = sorted(content_map.items(), key=lambda kv: kv[1], reverse=True)
        with open(content_rank_path, "w", encoding="utf-8") as f:
            for mid, score in sorted_content:
                f.write(f"{mid} : {score:.6f}\n")
        print(f"[INFO] 内容相似度排序已写入: {content_rank_path}")
    else:
        print("[INFO] 没有内容相似度数据，跳过 content_similarity_rank.txt")

    if style_map:
        style_rank_path = os.path.join(out_dir, "style_weighted_score_rank.txt")
        sorted_style = sorted(style_map.items(), key=lambda kv: kv[1], reverse=True)
        with open(style_rank_path, "w", encoding="utf-8") as f:
            for mid, score in sorted_style:
                f.write(f"{mid} : {score:.6f}\n")
        print(f"[INFO] 画风相似度排序已写入: {style_rank_path}")
    else:
        print("[INFO] 没有画风相似度数据，跳过 style_weighted_score_rank.txt")

    # ========== 根据前百分比 / model-id 白名单画每个 model_id 的九宫格 ==========
    # content 模式
    selected_content_ids: List[str] = []
    if model_id_whitelist is not None:
        selected_content_ids = [mid for mid in model_id_whitelist if mid in content_map]
        if selected_content_ids:
            print(
                f"[INFO] 将按 model-id txt 为 {len(selected_content_ids)} 个 content 模型生成九宫格"
            )
    elif args.content_top_pct is not None and content_map:
        pct = max(0.0, min(100.0, float(args.content_top_pct)))
        selected_content_ids = select_models_by_percent_range(
            content_map,
            (0.0, pct),
            descending=True,
        )
    if selected_content_ids:
        content_grid_dir = os.path.join(out_dir, "top_content_grids")
        os.makedirs(content_grid_dir, exist_ok=True)
        print(
            f"[INFO] 将为 {len(selected_content_ids)} 个 content 模型生成九宫格，输出到 {content_grid_dir}"
        )
        for mid in selected_content_ids:
            out_path = os.path.join(content_grid_dir, f"{mid}.png")
            build_single_model_grid(
                root=root,
                model_id=mid,
                mode="content",
                num_images=args.grid_num,
                out_path=out_path,
            )
    else:
        if args.content_top_pct is not None and model_id_whitelist is None:
            print(f"[INFO] content_top_pct={args.content_top_pct} 未选出任何模型，跳过 content 九宫格生成")

    # style 模式
    selected_style_ids: List[str] = []
    if model_id_whitelist is not None:
        selected_style_ids = [mid for mid in model_id_whitelist if mid in style_map]
        if selected_style_ids:
            print(
                f"[INFO] 将按 model-id txt 为 {len(selected_style_ids)} 个 style 模型生成九宫格"
            )
    elif args.style_top_pct is not None and style_map:
        pct = max(0.0, min(100.0, float(args.style_top_pct)))
        selected_style_ids = select_models_by_percent_range(
            style_map,
            (0.0, pct),
            descending=True,
        )
    if selected_style_ids:
        style_grid_dir = os.path.join(out_dir, "top_style_grids")
        os.makedirs(style_grid_dir, exist_ok=True)
        print(
            f"[INFO] 将为 {len(selected_style_ids)} 个 style 模型生成九宫格，输出到 {style_grid_dir}"
        )
        for mid in selected_style_ids:
            out_path = os.path.join(style_grid_dir, f"{mid}.png")
            build_single_model_grid(
                root=root,
                model_id=mid,
                mode="style",
                num_images=args.grid_num,
                out_path=out_path,
            )
    else:
        if args.style_top_pct is not None and model_id_whitelist is None:
            print(f"[INFO] style_top_pct={args.style_top_pct} 未选出任何模型，跳过 style 九宫格生成")


if __name__ == "__main__":
    main()

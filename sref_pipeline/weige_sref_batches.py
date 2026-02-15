#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import multiprocessing as mp
from io import BytesIO
from typing import List

from PIL import Image
from google import genai
from google.genai import types
from tqdm import tqdm

# ================== 配置 ==================

# 全局 client，用“懒加载”，保证每个进程里只初始化一次
_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            http_options={
                "api_version": "v1alpha",
                "base_url": "https://models-proxy.stepfun-inc.com/gemini",
            },
            api_key="ak-83d7efgh21i5jkl34mno90pqrs62tuv4k1",  # ⚠️ 实际使用建议改成环境变量
        )
    return _client


# 支持的图片后缀
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 按图标比例选 bucket
IMAGE_SIZE_BUCKETS = ["1:1", "3:4", "4:3", "9:16", "16:9"]


# ================== 工具函数 ==================
def parse_ratio(r: str) -> float:
    """把 '3:4' 变成浮点比例 3/4"""
    w, h = r.split(":")
    return float(w) / float(h)


def crop_to_ratio(img: Image.Image, target_ratio_str: str) -> Image.Image:
    """
    按目标宽高比居中裁剪（不缩放），主要用于图标，让输出图标保持类似构图比例。
    """
    target_ratio = parse_ratio(target_ratio_str)
    w, h = img.size
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-6:
        return img

    if current_ratio > target_ratio:
        # 图太宽 → 裁左右
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        # 图太高 → 裁上下
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)

    return img.crop(box)


def resize_to_match(img: Image.Image, target_size) -> Image.Image:
    """把图片 resize 到目标分辨率"""
    return img.resize(target_size, Image.LANCZOS)


def list_images(root: str) -> List[str]:
    """递归遍历 root 下所有图片路径"""
    results = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMG_EXTS:
                results.append(os.path.join(dirpath, name))
    results.sort()
    return results


# ================== Prompt（沿用你现在的版本） ==================
STYLE_PROMPT = """
You are an expert mobile icon designer.
There are two images:
- The FIRST image is an app icon demo. This is the content that must be preserved.
- The SECOND image is a phone wallpaper from the real world. This is ONLY a style reference.

Please edit ONLY the icon (first image) with the following constraints:
- Preserve the icon content: keep the main shapes, symbols, layout, silhouette and recognizability.
- Do NOT turn the icon into a realistic photo. It must remain a clean, graphic, vector-like app icon.
- Use the wallpaper ONLY to harmonize the icon with the wallpaper:
  - adapt the color palette, hue, saturation, and overall mood,
  - optionally adjust lighting direction and contrast to feel coherent with the wallpaper.
- Do NOT copy detailed objects or scenery from the wallpaper into the icon.
- Keep the background of the icon simple and uncluttered, suitable to sit on top of that wallpaper.

Output only the restyled icon.
"""


# ================== 核心处理函数 ==================
def process_pair(
    icon_path: str,
    wallpaper_path: str,
    out_content_dir: str,
    out_style_dir: str,
    out_generated_dir: str,
    overwrite: bool = False,
):
    """
    对单个 (icon_path, wallpaper_path) 组合调用 Gemini，
    文件命名规则：内容basename__风格basename（不含后缀）。
    """

    # 组合名：contentBasename__styleBasename
    content_stem = os.path.splitext(os.path.basename(icon_path))[0]
    style_stem = os.path.splitext(os.path.basename(wallpaper_path))[0]
    pair_id = f"{content_stem}__{style_stem}"

    # 生成图统一用 .png
    out_gen_path = os.path.join(out_generated_dir, f"{pair_id}.png")

    # ================== overwrite 检查 ==================
    if (not overwrite) and os.path.exists(out_gen_path):
        # 已生成过，直接跳过（避免重复请求模型）
        print(f"[SKIP] {pair_id} already exists, skip.")
        return pair_id

    # ========= 1. 读入 & 预处理图标 =========
    with Image.open(icon_path) as icon_img:
        icon_img = icon_img.convert("RGB")
        w, h = icon_img.size
        ratio = w / h
        image_aspect_ratio = min(
            IMAGE_SIZE_BUCKETS, key=lambda x: abs(parse_ratio(x) - ratio)
        )
        icon_cropped = crop_to_ratio(icon_img, image_aspect_ratio)

    # ========= 2. 读入 & 预处理壁纸 =========
    with Image.open(wallpaper_path) as wallpaper_img:
        wallpaper_img = wallpaper_img.convert("RGB")
        MAX_WALLPAPER_SIDE = 1600
        w, h = wallpaper_img.size
        if max(w, h) > MAX_WALLPAPER_SIDE:
            if w >= h:
                new_w = MAX_WALLPAPER_SIDE
                new_h = int(MAX_WALLPAPER_SIDE * h / w)
            else:
                new_h = MAX_WALLPAPER_SIDE
                new_w = int(MAX_WALLPAPER_SIDE * w / h)
            wallpaper_img = wallpaper_img.resize((new_w, new_h), Image.LANCZOS)

        # ========= 3. 调用 Gemini =========
        client = get_client()
        response = client.models.generate_content(
            model="gemini-3-pro-image-native",
            contents=[
                STYLE_PROMPT,
                icon_cropped,   # 第一张：内容图（图标）
                wallpaper_img,  # 第二张：风格参考（壁纸）
            ],
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                image_config=types.ImageConfig(
                    aspect_ratio=image_aspect_ratio,
                    image_size="1K",
                ),
            ),
        )

    # ========= 4. 提取图片结果 =========
    image_parts = [
        part for part in response.parts if getattr(part, "inline_data", None) is not None
    ]
    if not image_parts:
        print(f"[WARN] No image in response for pair {pair_id}")
        return None

    blob = image_parts[0].inline_data
    gen_img = Image.open(BytesIO(blob.data))

    # ========= 5. 保存三张图 =========
    # 内容图：用原内容图后缀
    content_ext = os.path.splitext(icon_path)[1].lower()
    if content_ext not in IMG_EXTS:
        content_ext = ".png"
    out_content_path = os.path.join(out_content_dir, f"{pair_id}{content_ext}")
    icon_cropped.save(out_content_path)

    # 风格图：用原风格图后缀
    style_ext = os.path.splitext(wallpaper_path)[1].lower()
    if style_ext not in IMG_EXTS:
        style_ext = ".png"
    out_style_path = os.path.join(out_style_dir, f"{pair_id}{style_ext}")
    wallpaper_img.save(out_style_path)

    # 生成图：统一 png
    gen_img.save(out_gen_path)

    print(f"[OK] pair_id={pair_id}")
    return pair_id


def worker_task(args):
    """给多进程池用的封装，args 是一个 tuple"""
    (
        icon_path,
        style_path,
        out_content_dir,
        out_style_dir,
        out_generated_dir,
        overwrite,
    ) = args
    try:
        return process_pair(
            icon_path,
            style_path,
            out_content_dir,
            out_style_dir,
            out_generated_dir,
            overwrite,
        )
    except Exception as e:
        pair_desc = f"{os.path.basename(icon_path)} __ {os.path.basename(style_path)}"
        print(f"[ERROR] Failed pair {pair_desc}: {e}")
        return None


# ================== 主程序 ==================
def main():
    parser = argparse.ArgumentParser(
        description="Batch SRef (multi-process): content icons × style images via Gemini"
    )
    parser.add_argument("--content-dir", required=True, help="内容图（图标）根目录，递归遍历")
    parser.add_argument("--style-dir", required=True, help="风格图（壁纸）根目录，递归遍历")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出根目录，内部会创建 content/style/generated 三个子目录",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["cartesian", "zip"],
        default="cartesian",
        help="cartesian: 所有内容×所有风格；zip: 按排序后一一配对",
    )
    parser.add_argument(
        "--max-content",
        type=int,
        default=None,
        help="最多使用前 N 张内容图（可选）",
    )
    parser.add_argument(
        "--max-style",
        type=int,
        default=None,
        help="最多使用前 N 张风格图（可选）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="多进程 worker 数量（默认 1，即单进程）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果设置，则对已有组合也重新生成；否则如果 generated 已存在则跳过",
    )

    args = parser.parse_args()

    content_paths = list_images(args.content_dir)
    style_paths = list_images(args.style_dir)

    if args.max_content is not None:
        content_paths = content_paths[: args.max_content]
    if args.max_style is not None:
        style_paths = style_paths[: args.max_style]

    if not content_paths:
        print("[ERROR] No content images found.")
        return
    if not style_paths:
        print("[ERROR] No style images found.")
        return

    # 输出目录
    out_content_dir = os.path.join(args.out_dir, "content")
    out_style_dir = os.path.join(args.out_dir, "style")
    out_generated_dir = os.path.join(args.out_dir, "generated")
    os.makedirs(out_content_dir, exist_ok=True)
    os.makedirs(out_style_dir, exist_ok=True)
    os.makedirs(out_generated_dir, exist_ok=True)

    # 组装任务列表
    tasks = []
    if args.pair_mode == "cartesian":
        print(
            f"[INFO] pair_mode=cartesian, {len(content_paths)} content × "
            f"{len(style_paths)} style → {len(content_paths) * len(style_paths)} pairs"
        )
        for c_path in content_paths:
            for s_path in style_paths:
                tasks.append(
                    (
                        c_path,
                        s_path,
                        out_content_dir,
                        out_style_dir,
                        out_generated_dir,
                        args.overwrite,
                    )
                )
    else:  # zip
        total = min(len(content_paths), len(style_paths))
        print(
            f"[INFO] pair_mode=zip, using first {total} pairs "
            f"(content & style sorted by path)"
        )
        for c_path, s_path in zip(content_paths, style_paths):
            tasks.append(
                (
                    c_path,
                    s_path,
                    out_content_dir,
                    out_style_dir,
                    out_generated_dir,
                    args.overwrite,
                )
            )

    # 多进程执行
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("[WARN] No pairs to process.")
        return

    num_workers = max(1, int(args.num_workers))

    print(f"[INFO] Start processing {total_tasks} pairs with {num_workers} workers.")

    with mp.Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(worker_task, tasks),
            total=total_tasks,
            desc="Processing pairs",
        ):
            pass


if __name__ == "__main__":
    main()

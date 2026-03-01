#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内容：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit \
  --jsons /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/dinov2_out.json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cas_out.json \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/vis_content \
  --long_side 512

风格：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit \
  --jsons /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/csd_out.json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/oneig_out.json \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/vis_style \
  --long_side 512

其他：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders  /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit \
  --jsons /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/clipcap_out.json  /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/laion_scores.json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/v25_scores.json \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/vis_other \
  --long_side 512 \
  --caption_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json

指令遵循可视化：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit \
  --jsons /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/follow_scores.json \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/vis_follow_new_1 \
  --long_side 512 \
  --caption_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json \
    /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/follow_scores.json \
    /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/follow_reasons.json

风格相似度可视化：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit \
  --jsons /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_style_descrete.json\
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/vis_style_vlm \
  --long_side 512 \
  --caption_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_style_reason_descrete.json

内容相似度可视化：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit \
  --jsons /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_content_descrete.json \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/vis_content_vlm \
  --long_side 512 \
    --caption_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_content_reason_descrete.json


"""
import argparse
import json
import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


def find_image_by_stem(folder: Path, stem: str) -> Optional[Path]:
    """Find image file in folder whose basename (stem) matches."""
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: scan folder (slower but robust)
    # Useful when extension is unknown or mixed-case
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem == stem:
            return p
    return None


def resize_long_side(img: Image.Image, long_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) == long_side:
        return img
    if w >= h:
        new_w = long_side
        new_h = int(round(h * (long_side / w)))
    else:
        new_h = long_side
        new_w = int(round(w * (long_side / h)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_font(preferred_size: int) -> ImageFont.FreeTypeFont:
    """
    Try to load a readable TrueType font. Fallback to default PIL bitmap font.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, preferred_size)
    return ImageFont.load_default()


def measure_multiline(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, spacing: int) -> Tuple[int, int]:
    # Pillow >= 8: multiline_textbbox is available
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h


def fit_font_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_w: int,
    max_h: int,
    start_size: int,
    min_size: int = 18,
) -> Tuple[ImageFont.ImageFont, int]:
    """
    Choose largest font size that fits in (max_w, max_h).
    """
    lo, hi = min_size, max(start_size, min_size)
    best_size = lo

    # If bitmap fallback font, just return it
    test_font = load_font(start_size)
    if not hasattr(test_font, "getbbox"):
        return test_font, 4

    spacing = max(4, start_size // 10)

    # Binary search
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(mid)
        spacing = max(4, mid // 10)
        tw, th = measure_multiline(draw, text, font, spacing)
        if tw <= max_w and th <= max_h:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    final_font = load_font(best_size)
    final_spacing = max(4, best_size // 10)
    return final_font, final_spacing


def concat_horizontally(imgs: List[Image.Image], bg=(255, 255, 255)) -> Image.Image:
    heights = [im.size[1] for im in imgs]
    widths = [im.size[0] for im in imgs]
    H = max(heights)
    W = sum(widths)

    canvas = Image.new("RGB", (W, H), bg)
    x = 0
    for im in imgs:
        y = (H - im.size[1]) // 2
        canvas.paste(im, (x, y))
        x += im.size[0]
    return canvas


def draw_center_text(
    img: Image.Image,
    lines: List[str],
    min_font: int = 18,
    start_font: Optional[int] = None,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    text = "\n".join(lines)

    W, H = img.size
    # Text box max size: keep margins
    box_w = int(W * 0.92)
    box_h = int(H * 0.80)

    if start_font is None:
        start_font = max(24, H // 18)  # big enough by default

    font, spacing = fit_font_to_box(draw, text, box_w, box_h, start_font, min_size=min_font)
    tw, th = measure_multiline(draw, text, font, spacing)

    x = (W - tw) // 2
    y = (H - th) // 2

    # Stroke for readability
    # Pillow supports stroke_width/stroke_fill in multiline_text
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255),
        spacing=spacing,
        align="center",
        stroke_width=max(2, (getattr(font, "size", 24) // 14)),
        stroke_fill=(0, 0, 0),
    )
    return img


def normalize_metric_value(name: str, value):
    if not isinstance(value, (int, float)):
        return value
    n = name.lower()
    if "csd" in n:
        v = (float(value) + 1.0) / 2.0
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v
    return value


def process_stem(payload):
    (
        stem,
        folder_paths,
        json_dicts,
        out_dir,
        long_side,
        bg,
        min_font,
        start_font,
        strict,
        caption_map,
    ) = payload
    folders = [Path(p) for p in folder_paths]
    out_dir = Path(out_dir)
    img_paths = []
    missing_folder = False
    for fd in folders:
        ip = find_image_by_stem(fd, stem)
        if ip is None:
            missing_folder = True
            break
        img_paths.append(ip)

    lines = []
    missing_json = False
    for jname, jd in json_dicts:
        if stem not in jd:
            missing_json = True
            val = "N/A"
        else:
            val = normalize_metric_value(jname, jd[stem])
        lines.append(f"{jname}: {val}")

    if strict and (missing_folder or missing_json):
        return ("skipped", stem)
    if missing_folder:
        return ("skipped", stem)

    imgs = []
    for ip in img_paths:
        im = Image.open(ip).convert("RGB")
        im = resize_long_side(im, long_side)
        imgs.append(im)

    canvas = concat_horizontally(imgs, bg=bg)
    canvas = draw_center_text(canvas, lines, min_font=min_font, start_font=start_font)
    out_path = out_dir / f"{stem}.png"
    canvas.save(out_path)
    if isinstance(caption_map, dict) and stem in caption_map:
        txt_path = out_dir / f"{stem}.txt"
        val = caption_map.get(stem, "")
        if isinstance(val, list):
            val = "\n".join(str(x) for x in val)
        elif not isinstance(val, str):
            val = str(val)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(val)
    return ("done", stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs="+", required=True, help="Image folders (>=2).")
    ap.add_argument("--jsons", nargs="+", required=True, help="One or more json files: {stem: value}.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--long_side", type=int, default=512, help="Resize long side for each image (default: 512).")
    ap.add_argument("--bg", default="white", choices=["white", "black"], help="Background color for padding.")
    ap.add_argument("--min_font", type=int, default=18, help="Minimum font size (default: 18).")
    ap.add_argument("--start_font", type=int, default=0, help="Start font size (0 means auto).")
    ap.add_argument("--strict", action="store_true", help="If set, skip stems missing in any folder or any json.")
    ap.add_argument("--num_procs", type=int, default=16, help="Number of processes (default: 4).")
    ap.add_argument("--caption_json", nargs="*", default=[], help="Optional caption json path(s) {stem: caption}. If set, write txt.")
    args = ap.parse_args()

    folders = [Path(p) for p in args.folders]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg = (255, 255, 255) if args.bg == "white" else (0, 0, 0)

    # Load jsons
    json_dicts: List[Tuple[str, Dict[str, object]]] = []
    all_keys = set()

    for jp in args.jsons:
        p = Path(jp)
        name = p.stem
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError(f"{jp} is not a dict json.")
        json_dicts.append((name, d))
        all_keys.update(d.keys())

    caption_map: Dict[str, List[object]] = {}
    if args.caption_json:
        for jp in args.caption_json:
            p = Path(jp)
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                raise ValueError(f"{jp} is not a dict json.")
            for k, v in d.items():
                caption_map.setdefault(k, []).append(v)

    # Iterate keys (sorted for determinism)
    keys = sorted(all_keys)

    start_font = None if args.start_font <= 0 else args.start_font

    skipped = 0
    done = 0

    payloads = [
        (
            stem,
            [str(p) for p in folders],
            json_dicts,
            str(out_dir),
            args.long_side,
            bg,
            args.min_font,
            start_font,
            args.strict,
            caption_map,
        )
        for stem in keys
    ]

    num_procs = max(1, int(args.num_procs))
    if num_procs == 1:
        for payload in tqdm(payloads, total=len(payloads), unit="img"):
            status, _ = process_stem(payload)
            if status == "done":
                done += 1
            else:
                skipped += 1
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_procs) as pool:
            for status, _ in tqdm(pool.imap_unordered(process_stem, payloads), total=len(payloads), unit="img"):
                if status == "done":
                    done += 1
                else:
                    skipped += 1

    print(f"[OK] done={done}, skipped={skipped}, out_dir={out_dir}")


if __name__ == "__main__":
    main()

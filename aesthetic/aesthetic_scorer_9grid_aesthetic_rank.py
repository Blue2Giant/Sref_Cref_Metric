#!/usr/bin/env python3

import os
import argparse
import json
from typing import List, Tuple

import torch
from PIL import Image
from transformers import CLIPProcessor
from aesthetic_scorer import AestheticScorer


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_NAME = "/mnt/jfs/model_zoo/aesthetic-scorer/"
MODEL_WEIGHTS = "/mnt/jfs/model_zoo/aesthetic-scorer/model.pt"

_PROCESSOR = None
_MODEL = None
_DEVICE = None


def is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def list_images(dir_path: str) -> List[str]:
    names = sorted(os.listdir(dir_path))
    return [os.path.join(dir_path, n) for n in names if is_image_file(n)]


def split_3x3(img: Image.Image) -> List[Image.Image]:
    w, h = img.size
    tile_w = w // 3
    tile_h = h // 3
    crops: List[Image.Image] = []
    for row in range(3):
        for col in range(3):
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w
            lower = upper + tile_h
            crops.append(img.crop((left, upper, right, lower)))
    return crops


def ensure_processor() -> CLIPProcessor:
    global _PROCESSOR
    if _PROCESSOR is not None:
        return _PROCESSOR
    _PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _PROCESSOR


def ensure_model() -> torch.nn.Module:
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _DEVICE = device
    obj = torch.load(MODEL_WEIGHTS, map_location=device)
    if isinstance(obj, dict):
        model = AestheticScorer()
        model.load_state_dict(obj)
    else:
        model = obj
    model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    _MODEL = model
    return _MODEL


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    ensure_model()
    return _DEVICE


def score_tile_overall(tile: Image.Image) -> float:
    processor = ensure_processor()
    model = ensure_model()
    device = get_device()

    inputs = processor(images=tile, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        scores = model(inputs)

    if isinstance(scores, torch.Tensor):
        if scores.ndimension() == 0:
            return float(scores.detach().cpu().item())
        if scores.ndimension() >= 1:
            vec = scores.view(-1)
            if vec.numel() == 0:
                return 0.0
            return float(vec[0].detach().cpu().item())

    if isinstance(scores, (list, tuple)) and scores:
        first = scores[0]
        if isinstance(first, torch.Tensor):
            v = first.view(-1)
            if v.numel() == 0:
                return 0.0
            return float(v[0].detach().cpu().item())
        if isinstance(first, (int, float)):
            return float(first)

    if isinstance(scores, (int, float)):
        return float(scores)

    return 0.0


def score_9grid_image(path: str) -> Tuple[float, int]:
    img = Image.open(path).convert("RGB")
    tiles = split_3x3(img)

    values: List[float] = []
    for t in tiles:
        v = score_tile_overall(t)
        values.append(float(v))

    if not values:
        return 0.0, 0

    avg = float(sum(values) / len(values))
    return avg, len(values)


def main():
    parser = argparse.ArgumentParser(
        description="使用 aesthetic-scorer 对九宫格图片拆分后逐张打分，只用 Overall 维度求平均并按平均分排序"
    )
    parser.add_argument("--input-dir", required=True, help="包含九宫格图片的目录")
    parser.add_argument("--output-txt", default="", help="可选：输出排序后的 basename 列表 txt")
    parser.add_argument("--output-json", required=True, help="输出排序后的 basename->平均分 JSON")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_txt = args.output_txt
    output_json = args.output_json

    if not os.path.isdir(input_dir):
        raise SystemExit(f"input-dir 不存在或不是目录: {input_dir}")

    img_paths = list_images(input_dir)
    if not img_paths:
        raise SystemExit(f"目录下没有图片文件: {input_dir}")

    results: List[Tuple[str, float]] = []

    for p in img_paths:
        try:
            avg_score, n_tiles = score_9grid_image(p)
        except Exception as e:
            print(f"[WARN] 处理失败: {p}: {e}")
            continue
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        results.append((stem, avg_score))
        print(f"[INFO] {base}: avg_score={avg_score:.4f}")

    if not results:
        raise SystemExit("没有任何有效的打分结果")

    results.sort(key=lambda x: x[1], reverse=True)

    if output_txt:
        out_dir_txt = os.path.dirname(output_txt)
        if out_dir_txt:
            os.makedirs(out_dir_txt, exist_ok=True)
        with open(output_txt, "w", encoding="utf-8") as f:
            for stem, _score in results:
                f.write(stem + "\n")
        print(f"[DONE] 写入排序结果 txt: {output_txt}")

    data = {stem: score for stem, score in results}
    out_dir_json = os.path.dirname(output_json)
    if out_dir_json:
        os.makedirs(out_dir_json, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 写入排序结果 json: {output_json}")


if __name__ == "__main__":
    main()

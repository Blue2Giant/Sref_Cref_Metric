#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/benchmark_metrics/benchmark_metrics/csd_cosine_demo.py \
  --image_a /data/benchmark_metrics/assets/stylized.png\
  --image_b /data/benchmark_metrics/assets/style.webp \
  --model_path style_models/checkpoint.pth \
  --clip_model_path /mnt/jfs/model_zoo/open_clip/ViT-L-14-openai.pt \
  --device cuda:0
"""
import argparse
import numpy as np
from PIL import Image
from csd_utils import CSDStyleEmbedding


def load_rgb_resize(path: str, size: int = 512) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((size, size))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_a", required=True)
    ap.add_argument("--image_b", required=True)
    ap.add_argument("--model_path", default="style_models/checkpoint.pth")
    ap.add_argument("--clip_model_path", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    clip_model_path = args.clip_model_path.strip() or None
    encoder = CSDStyleEmbedding(model_path=args.model_path, device=args.device, clip_model_path=clip_model_path)
    img_a = load_rgb_resize(args.image_a, args.size)
    img_b = load_rgb_resize(args.image_b, args.size)
    embed_a = encoder.get_style_embedding(img_a)
    embed_b = encoder.get_style_embedding(img_b)
    score = cosine_similarity(np.array(embed_a), np.array(embed_b))
    print(score)


if __name__ == "__main__":
    main()

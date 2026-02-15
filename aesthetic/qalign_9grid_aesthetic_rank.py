#!/usr/bin/env python3

import os
import argparse
import json
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_NAME = "/mnt/jfs/model_zoo/one-align"


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


def ensure_model() -> AutoModelForCausalLM:
    global _MODEL
    if "_MODEL" in globals() and _MODEL is not None:
        return _MODEL
    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return _MODEL


def scores_to_list(scores) -> List[float]:
    if isinstance(scores, torch.Tensor):
        arr = scores.detach().cpu().view(-1).tolist()
        return [float(x) for x in arr]
    if isinstance(scores, (list, tuple)):
        flat: List[float] = []
        for x in scores:
            if isinstance(x, torch.Tensor):
                flat.extend(scores_to_list(x))
            elif isinstance(x, (int, float)):
                flat.append(float(x))
        return flat
    if isinstance(scores, (int, float)):
        return [float(scores)]
    return []


def score_9grid_image(path: str) -> Tuple[float, int]:
    img = Image.open(path).convert("RGB")
    tiles = split_3x3(img)
    model = ensure_model()
    
    s_list: List[float] = []
    for t in tiles:
        # 用户反馈批量传入会报错 Cache 未定义，改为单张处理
        # 注意：model.score 接口通常要求输入是 list
        ret = model.score([t], task_="aesthetics", input_="image")
        s_list.extend(scores_to_list(ret))

    if not s_list:
        return 0.0, 0
    avg = float(sum(s_list) / len(s_list))
    return avg, len(s_list)


def main():
    parser = argparse.ArgumentParser(
        description="对目录下九宫格图片拆分为9张并用 Q-Align aesthetics 打分求平均，根据平均分从高到低输出结果"
    )
    parser.add_argument("--input-dir", required=True, help="包含九宫格图片的目录")
    parser.add_argument("--output-txt", default="", help="可选：输出排序后的 basename 列表 txt")
    parser.add_argument("--output-json", required=True, help="输出排序后的 basename->平均分 JSON")
    parser.add_argument("--filter-txt", default="", help="可选：仅处理该txt中列出的 model_id (每行一个)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_txt = args.output_txt
    output_json = args.output_json
    filter_txt = args.filter_txt

    existing_data = None
    existing_keys = None
    if os.path.isfile(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, dict):
                existing_keys = set(str(k) for k in existing_data.keys())
                print(f"[INFO] 已加载已有结果 json: {output_json}，包含 {len(existing_keys)} 个 key")
            else:
                existing_data = None
                existing_keys = None
        except Exception as e:
            print(f"[WARN] 读取已有结果 json 失败: {output_json}: {e}")
            existing_data = None
            existing_keys = None

    if not os.path.isdir(input_dir):
        raise SystemExit(f"input-dir 不存在或不是目录: {input_dir}")

    img_paths = list_images(input_dir)
    if not img_paths:
        raise SystemExit(f"目录下没有图片文件: {input_dir}")

    # Load filter list if provided
    allowed_ids = None
    if filter_txt:
        if os.path.isfile(filter_txt):
            with open(filter_txt, "r", encoding="utf-8") as f:
                allowed_ids = set(line.strip() for line in f if line.strip())
            print(f"[INFO] 已加载过滤列表，包含 {len(allowed_ids)} 个 ID")
        else:
            print(f"[WARN] 指定的 filter-txt 不存在: {filter_txt}，将跳过过滤")

    results: List[Tuple[str, float]] = []

    for p in img_paths:
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)

        if allowed_ids is not None:
            if stem not in allowed_ids:
                continue
        if existing_keys is not None and stem in existing_keys:
            print(f"[INFO] 跳过已存在结果: {stem}")
            continue

        try:
            avg_score, n_tiles = score_9grid_image(p)
        except Exception as e:
            print(f"[WARN] 处理失败: {p}: {e}")
            continue
        
        results.append((stem, avg_score))
        print(f"[INFO] {base}: avg_score={avg_score:.4f}")

    if not results:
        if existing_data:
            print("[INFO] 没有新的打分结果，保持原有 json 不变")
            return
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

    data = {}
    if isinstance(existing_data, dict):
        for k, v in existing_data.items():
            data[str(k)] = v
    for stem, score in results:
        data[stem] = str(score)
    out_dir_json = os.path.dirname(output_json)
    if out_dir_json:
        os.makedirs(out_dir_json, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 写入排序结果 json: {output_json}")


if __name__ == "__main__":
    main()

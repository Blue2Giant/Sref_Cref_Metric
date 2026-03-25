#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Qwen-Image-Edit-2511 单样本最小推理（对齐Qwen_2511_demo输入风格）")
    parser.add_argument("--prompts_json", required=True, help="id->prompt 的json文件")
    parser.add_argument("--cref_dir", required=True, help="内容参考图目录，文件名应为 {id}.png")
    parser.add_argument("--sref_dir", required=True, help="风格参考图目录，文件名应为 {id}.png")
    parser.add_argument("--out_dir", required=True, help="输出目录，保存 {id}.png")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--gpus", default="0", help='如 "0" 或 "0,1,2"，本脚本仅使用第一个GPU')
    parser.add_argument("--key_txt", required=True, help="txt文件，第一行非空字符串作为唯一key")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_prompts(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"prompts_json 结构错误: {path}")
    return {str(k): str(v) for k, v in data.items()}

#返回一个key的列表
def read_single_key(key_txt: str) -> list:
    with open(key_txt, "r", encoding="utf-8") as f:
        keys = []
        for line in f:
            s = (line or "").strip()
            if s:
                keys.append(s)
    return keys


def main():
    args = parse_args()
    gpu_list = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
    gpu = gpu_list[0] if gpu_list else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=False)

    prompts = load_prompts(args.prompts_json)
    keys_prompts = read_single_key(args.key_txt)
    prompt2gen = []
    cref_path2gen=[]
    sref_path2gen=[]
    for key in keys_prompts:
        if key in prompts.keys():
            prompt2gen.append(prompts[key])
            cref_path2gen.append(Path(args.cref_dir) / f"{key}.png")
            sref_path2gen.append(Path(args.sref_dir) / f"{key}.png")
            continue
        else:
            print(f"[SKIP] key不在prompts_json中，已跳过: {key}")

    for prompt, cref_path, sref_path in zip(prompt2gen, cref_path2gen, sref_path2gen):
        if not cref_path.exists():
            raise RuntimeError(f"cref不存在: {cref_path}")
        if not sref_path.exists():
            raise RuntimeError(f"sref不存在: {sref_path}")

        cref = load_rgb(str(cref_path))
        sref = load_rgb(str(sref_path))
        images = [cref, sref]

        generator = torch.Generator(device=device).manual_seed(int(args.seed))
        with torch.inference_mode():
            out = pipe(
                image=images,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                width=cref.size[0],
                height=cref.size[1],
                num_inference_steps=int(args.steps),
                true_cfg_scale=float(args.true_cfg_scale),
                generator=generator,
            ).images[0]

        out_path = Path(args.out_dir) / f"{key}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path)
        print(f"[DONE] key={key} saved: {out_path}")


if __name__ == "__main__":
    main()

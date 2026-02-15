#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import argparse
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
)
from typing import Optional
from PIL import Image, UnidentifiedImageError
import json
from megfile.smart import (
    smart_open as mopen,
    smart_listdir,
    smart_exists,
    smart_scandir,
    smart_makedirs,
    smart_isdir,
)
import io
# 只认为这些扩展名是图片
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 本地模型路径
DINO_MODEL_NAME = "/mnt/jfs/model_zoo/dinov2-base"
CLIP_MODEL_NAME = "/mnt/jfs/model_zoo/clip-vit-large-patch14"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 工具函数 =====================
def load_image(path: str) -> Image.Image:
    """读取并转成 RGB。"""
    return Image.open(path).convert("RGB")


def cosine_similarity(f1: torch.Tensor, f2: torch.Tensor) -> float:
    """计算两个特征的余弦相似度。"""
    f1 = F.normalize(f1.view(1, -1), p=2, dim=-1)
    f2 = F.normalize(f2.view(1, -1), p=2, dim=-1)
    return float(F.cosine_similarity(f1, f2, dim=-1).item())


def list_images_recursive(root: Path) -> List[Path]:
    """递归列出 root 下所有图片文件。"""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


# ===================== 特征提取相关 =====================
def load_backend(backend: str):
    """
    根据 backend 加载对应的模型和 processor。

    返回:
        backend (str),
        processor,
        model
    """
    if backend == "dino":
        print(f"加载 DINOv2 模型: {DINO_MODEL_NAME}")
        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device)
        model.eval()
        return backend, processor, model

    elif backend == "clip":
        print(f"加载 CLIP 模型: {CLIP_MODEL_NAME}")
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        model.eval()
        return backend, processor, model

    else:
        raise ValueError(f"未知 backend: {backend}")


def extract_feature(img: Image.Image, backend: str, processor, model) -> torch.Tensor:
    """
    统一的特征提取接口：
    - backend == "dino": DINOv2 CLS 特征
    - backend == "clip": CLIP 图像特征 (get_image_features)
    返回 [D]，已经 L2 归一化。
    """
    if backend == "dino":
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs[0]  # [1, tokens, D]
        cls_feat = last_hidden_states[:, 0, :]  # CLS token
        feat = F.normalize(cls_feat, p=2, dim=-1)  # [1, D]
        return feat[0]

    elif backend == "clip":
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_feats = model.get_image_features(**inputs)  # [1, D]
        feat = F.normalize(image_feats, p=2, dim=-1)  # [1, D]
        return feat[0]

    else:
        raise ValueError(f"未知 backend: {backend}")




def main():
    parser = argparse.ArgumentParser(
        description="用 DINO/CLIP 提取内容特征：probe vs 文件夹中所有图片，逐张算相似度并取平均"
    )
    parser.add_argument("--gallery_dir", required=True, help="待比较图片所在文件夹（递归搜索）")
    parser.add_argument("--probe_image", required=True, help="要与文件夹比较的单张图片路径")
    parser.add_argument(
        "--backend",
        choices=["dino", "clip"],
        default="dino",
        help="内容特征后端，可选 dino 或 clip，默认 dino",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="可选：把结果写入 JSON 文件路径（不指定则只打印到控制台）",
    )
    args = parser.parse_args()

    gallery_dir = Path(args.gallery_dir)
    if not gallery_dir.is_dir():
        raise SystemExit(f"gallery_dir 不是目录或不存在: {gallery_dir}")

    probe_path = Path(args.probe_image)
    if not probe_path.is_file():
        raise SystemExit(f"probe_image 不存在: {probe_path}")

    backend, processor, model = load_backend(args.backend)

    # 1. 预先算好 probe 的特征
    try:
        probe_img = load_image(str(probe_path))
    except (UnidentifiedImageError, OSError) as e:
        raise SystemExit(f"无法读取 probe_image: {probe_path} | {e}")

    probe_feat = extract_feature(probe_img, backend, processor, model)

    # 2. 遍历文件夹中的所有图片，逐张算与 probe 的相似度
    gallery_imgs = list_images_recursive(gallery_dir)
    if not gallery_imgs:
        raise SystemExit(f"在 {gallery_dir} 下没有找到图片")

    # 若 probe 本身就在目录里，为了避免 trivially 1.0，相同路径可以跳过
    probe_abs = probe_path.resolve()
    sims: Dict[str, float] = {}

    for img_path in gallery_imgs:
        if img_path.resolve() == probe_abs:
            print(f"[SKIP] 跳过与 probe 相同的文件: {img_path}")
            continue
        try:
            img = load_image(str(img_path))
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] 跳过坏图 {img_path}: {e}")
            continue

        feat = extract_feature(img, backend, processor, model)
        sim = cosine_similarity(probe_feat, feat)
        sims[str(img_path)] = sim
        print(f"[SIM] {img_path}  ->  {sim:.4f}")

    if not sims:
        raise SystemExit("没有任何有效 gallery 图片参与计算，相似度结果为空")

    # 3. 对所有相似度求平均
    mean_sim = sum(sims.values()) / len(sims)
    print("======================================")
    print(f"backend = {backend}")
    print(f"probe_image = {probe_path}")
    print(f"gallery_dir = {gallery_dir}")
    print(f"有效 gallery 图片数 = {len(sims)}")
    print(f"平均相似度 = {mean_sim:.4f}")
    print("======================================")

    # 4. 可选：导出 JSON
    if args.output_json:
        out = {
            "backend": backend,
            "probe_image": str(probe_path),
            "gallery_dir": str(gallery_dir),
            "num_gallery_images": len(sims),
            "per_image_similarity": sims,  # {路径: 相似度}
            "mean_similarity": mean_sim,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[DONE] 结果已写入 JSON: {out_path}")

def compute_gallery_similarity(
    gallery_dir: str,
    probe_image: str,
    backend: str = "dino",
    output_json: Optional[str] = None,
) -> Dict:
    """
    用 DINO/CLIP 提取内容特征：probe 与 gallery 中所有图片计算相似度并求平均。

    参数：
        gallery_dir (str): 需要遍历比对的图片文件夹（递归搜索）
        probe_image (str): 探针图片路径
        backend (str): "dino" 或 "clip"
        output_json (str or None): 可选，将结果保存为 JSON 文件

    返回：
        Dict: 包含 backend、probe、gallery、per-image 相似度、平均相似度
    """

    gallery_dir = Path(gallery_dir)
    if not gallery_dir.is_dir():
        raise ValueError(f"gallery_dir 不是目录或不存在: {gallery_dir}")

    probe_path = Path(probe_image)
    if not probe_path.is_file():
        raise ValueError(f"probe_image 不存在: {probe_path}")

    # 加载 backend
    backend_name, processor, model = load_backend(backend)

    # 加载 probe 特征
    try:
        probe_img = load_image(str(probe_path))
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"无法读取 probe_image: {probe_path} | {e}")

    probe_feat = extract_feature(probe_img, backend_name, processor, model)

    # 遍历所有图片
    gallery_imgs = list_images_recursive(gallery_dir)
    if not gallery_imgs:
        raise RuntimeError(f"在 {gallery_dir} 下没有找到任何图片")

    probe_abs = probe_path.resolve()
    sims: Dict[str, float] = {}

    for img_path in gallery_imgs:
        if img_path.resolve() == probe_abs:
            print(f"[SKIP] 跳过与 probe 相同的文件: {img_path}")
            continue
        
        try:
            img = load_image(str(img_path))
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] 跳过坏图 {img_path}: {e}")
            continue

        feat = extract_feature(img, backend_name, processor, model)
        sim = cosine_similarity(probe_feat, feat)
        sims[str(img_path)] = sim
        print(f"[SIM] {img_path}  ->  {sim:.4f}")

    if not sims:
        raise RuntimeError("没有任何有效 gallery 图片参与计算，相似度结果为空")

    # 计算平均相似度
    mean_sim = sum(sims.values()) / len(sims)

    print("======================================")
    print(f"backend = {backend_name}")
    print(f"probe_image = {probe_path}")
    print(f"gallery_dir = {gallery_dir}")
    print(f"有效 gallery 图片数 = {len(sims)}")
    print(f"平均相似度 = {mean_sim:.4f}")
    print("======================================")

    result = {
        "backend": backend_name,
        "probe_image": str(probe_path),
        "gallery_dir": str(gallery_dir),
        "num_gallery_images": len(sims),
        "per_image_similarity": sims,
        "mean_similarity": mean_sim,
    }

    # 可选导出 JSON
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[DONE] 结果已写入 JSON: {out_path}")

    return result

def is_remote_path_megfile(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("oss://")

def list_images_recursive_megfile(root: str) -> List[str]:
    """
    递归列出 root 下所有图片文件。
    支持本地路径和桶路径。
    """
    paths: List[str] = []

    if is_remote_path_megfile(root):
        # megfile 递归扫描
        stack = [root.rstrip("/")]
        while stack:
            cur = stack.pop()
            try:
                for entry in smart_scandir(cur):
                    try:
                        if entry.is_dir():
                            stack.append(entry.path)
                        else:
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in IMG_EXTS:
                                paths.append(entry.path)
                    except Exception:
                        continue
            except FileNotFoundError:
                continue
    else:
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        for p in root_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))

    paths.sort()
    return paths


def dir_exists_megfile(path: str) -> bool:
    """判断目录是否存在（本地 / 桶）。"""
    if is_remote_path_megfile(path):
        return smart_exists(path) and smart_isdir(path)
    return os.path.isdir(path)

def load_image(path: str) -> Image.Image:
    """本地路径直接读；桶路径用 megfile 读二进制再丢给 PIL。"""
    if is_remote_path_megfile(path):
        with mopen(path, "rb") as f:
            data = f.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        return Image.open(path).convert("RGB")

def iter_model_dirs_megfile(root: str) -> List[str]:
    """
    枚举 root 下的所有一级子目录（model_id 目录）。
    支持本地和桶。
    """
    dirs: List[str] = []

    if is_remote_path_megfile(root):
        try:
            for entry in smart_scandir(root):
                try:
                    if entry.is_dir():
                        dirs.append(entry.path)
                except Exception:
                    continue
        except FileNotFoundError:
            return []
    else:
        path_root = Path(root)
        if not path_root.is_dir():
            return []
        for p in path_root.iterdir():
            if p.is_dir():
                dirs.append(str(p))

    dirs.sort()
    return dirs
def compute_gallery_similarity_megfile(
    gallery_dir: str,
    probe_image: str,
    backend: str = "dino",
    output_json: Optional[str] = None,
    processor=None,
    model=None,
    verbose: bool = True,
) -> Dict:
    """
    用 DINO/CLIP 提取内容特征：probe 与 gallery 中所有图片计算相似度并求平均。

    参数：
        gallery_dir (str): 需要遍历比对的图片文件夹（递归搜索，本地或桶路径）
        probe_image (str): 探针图片路径（本地或桶路径）
        backend (str): "dino" 或 "clip"
        output_json (str or None): 可选，将结果保存为 JSON 文件（本地或桶路径）
        processor, model: 可选，已加载好的后端，传入就不会重复 load_backend
        verbose (bool): 是否在控制台打印详细日志

    返回：
        Dict: 包含 backend、probe、gallery、per-image 相似度、平均相似度
    """

    # --- 检查路径 ---
    if is_remote_path_megfile(gallery_dir):
        if not smart_exists(gallery_dir) or not smart_isdir(gallery_dir):
            raise ValueError(f"gallery_dir 不是目录或不存在: {gallery_dir}")
    else:
        g_path = Path(gallery_dir)
        if not g_path.is_dir():
            raise ValueError(f"gallery_dir 不是目录或不存在: {gallery_dir}")

    if is_remote_path_megfile(probe_image):
        if not smart_exists(probe_image):
            raise ValueError(f"probe_image 不存在: {probe_image}")
    else:
        p_path = Path(probe_image)
        if not p_path.is_file():
            raise ValueError(f"probe_image 不存在: {probe_image}")

    # --- 加载 backend（优先复用已有的 processor/model） ---
    if processor is None or model is None:
        backend_name, processor, model = load_backend(backend)
    else:
        backend_name = backend

    # --- 加载 probe 特征 ---
    try:
        probe_img = load_image(probe_image)
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"无法读取 probe_image: {probe_image} | {e}")

    probe_feat = extract_feature(probe_img, backend_name, processor, model)

    # --- 遍历 gallery ---
    gallery_imgs = list_images_recursive_megfile(gallery_dir)
    if not gallery_imgs:
        raise RuntimeError(f"在 {gallery_dir} 下没有找到任何图片")

    # probe 自身若在 gallery 中则跳过
    # 对于桶路径就按字符串比较即可；本地路径统一成绝对路径比较
    if not is_remote_path_megfile(probe_image):
        probe_abs = str(Path(probe_image).resolve())
    else:
        probe_abs = probe_image

    sims: Dict[str, float] = {}

    for img_path in gallery_imgs:
        same_flag = False
        if is_remote_path_megfile(img_path) or is_remote_path_megfile(probe_image):
            if img_path == probe_abs:
                same_flag = True
        else:
            if str(Path(img_path).resolve()) == probe_abs:
                same_flag = True

        if same_flag:
            if verbose:
                print(f"[SKIP] 跳过与 probe 相同的文件: {img_path}")
            continue

        try:
            img = load_image(img_path)
        except (UnidentifiedImageError, OSError) as e:
            if verbose:
                print(f"[WARN] 跳过坏图 {img_path}: {e}")
            continue

        feat = extract_feature(img, backend_name, processor, model)
        sim = cosine_similarity(probe_feat, feat)
        sims[img_path] = sim
        if verbose:
            print(f"[SIM] {img_path}  ->  {sim:.4f}")

    if not sims:
        raise RuntimeError("没有任何有效 gallery 图片参与计算，相似度结果为空")

    # --- 计算平均相似度 ---
    mean_sim = sum(sims.values()) / len(sims)

    if verbose:
        print("======================================")
        print(f"backend = {backend_name}")
        print(f"probe_image = {probe_image}")
        print(f"gallery_dir = {gallery_dir}")
        print(f"有效 gallery 图片数 = {len(sims)}")
        print(f"平均相似度 = {mean_sim:.4f}")
        print("======================================")

    result = {
        "backend": backend_name,
        "probe_image": probe_image,
        "gallery_dir": gallery_dir,
        "num_gallery_images": len(sims),
        "per_image_similarity": sims,
        "mean_similarity": mean_sim,
    }

    # --- 可选导出 JSON（本地 / 桶） ---
    if output_json:
        out_dir = os.path.dirname(output_json)
        if out_dir:
            smart_makedirs(out_dir, exist_ok=True)
        with mopen(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"[DONE] 结果已写入 JSON: {output_json}")

    return result
if __name__ == "__main__":
    main()

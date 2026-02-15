#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多进程版 LoRA 三元组图片风格相似度工具（多后端，支持桶路径，一块 GPU 一个进程可选）。

目录结构说明
===========

1. 三元组目录（--root）：
   --root
     ├── <style_model_id>__<content_model_id>/
     │     ├── content_100/       (或其它以 content_ 开头的目录，可选)
     │     ├── style_100/         (或其它以 style_   开头的目录，可选)
     │     └── two_100/           (或其它以 two_     开头的目录，可选)
     ├── ...

   对于每个 <style_model_id>__<content_model_id> 目录，会对其中
   style_/content_/two_ 子目录下的所有图片计算风格相似度。

2. 风格基准目录（--style-root）：
   --style-root
     ├── <model_id>/
     │     ├── style_100/ ...
     │     ├── ...
     ├── ...

   对于每个 style_model_id，到 --style-root/<style_model_id>/ 下寻找
   第一个以 "style_" 开头的子目录，将其中所有图片特征取均值，得到
   每个 encoder 的风格均值向量，并缓存为
       <encoder_name>_mean.pth
   例如：
       siglip_mean.pth, styleshot_mean.pth, ...

3. model_id 图像列表（--id-list，可选）：
   txt 文件，每行可以是：
       1234567
       1234567.png
       s3://.../some/path/1234567.png
   脚本会取每一行的 basename 去掉扩展名，得到 style_model_id。
   只处理这些 style_model_id 对应的
   <style_model_id>__<content_model_id> 目录；不传则处理所有。

4. 输出：

   4.1 全局输出（--output-json）：
   {
     "pair_root": "...",
     "style_root": "...",
     "pt_style": ["siglip", "oneig", ...],
     "encoder_weights_used": {"siglip": 0.5, "oneig": 0.5},
     "scores": {
        "s3://.../1116635__1251080/style_100/00001.png": 0.91,
        "s3://.../1116635__1251080/two_100/00001.png":   0.88,
        ...
     }
   }

   4.2 每个三元组目录下的局部输出：
       <style_model_id>__<content_model_id>/triplet_style_similarity.json

   结构示例：
   {
     "pair_dir": "s3://.../1116635__1251080",
     "style_id": "1116635",
     "num_images": 300,
     "pt_style": ["siglip", "oneig"],
     "encoder_weights_used": {"siglip": 0.5, "oneig": 0.5},
     "scores": {
       "s3://.../1116635__1251080/style_100/00001.png": 0.91,
       "s3://.../1116635__1251080/two_100/00001.png":   0.88
     }
   }
"""

import os
import re
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import multiprocessing as mp

from PIL import Image
import torch
import torch.nn as nn

# ---------- megfile 相关 ----------
from megfile.smart import (
    smart_scandir,
    smart_exists,
    smart_isdir,
    smart_open as mopen,
    smart_makedirs,
)

# ---------- 一些可能用到的外部依赖 ----------
try:
    from transformers import AutoModel, AutoImageProcessor, AutoProcessor
except Exception:
    AutoModel = None
    AutoImageProcessor = None
    AutoProcessor = None

try:
    from torchvision import transforms
    from torchvision.models import vgg16, VGG16_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
except Exception:
    transforms = None
    vgg16 = None
    VGG16_Weights = None
    create_feature_extractor = None

try:
    from CSD.model import CSD_CLIP
    from CSD.utils import has_batchnorms, convert_state_dict
    from CSD.loss_utils import transforms_branch0
except Exception:
    CSD_CLIP = None
    has_batchnorms = None
    convert_state_dict = None
    transforms_branch0 = None

try:
    from models import clip as local_clip
except Exception:
    local_clip = None
    try:
        import clip as pip_clip
    except Exception:
        pip_clip = None

try:
    from models import dino_vits, moco_vits
except Exception:
    dino_vits = None
    moco_vits = None

try:
    from ip_adapter.style_encoder import Style_Aware_Encoder
    from transformers import CLIPVisionModelWithProjection
except Exception:
    Style_Aware_Encoder = None
    CLIPVisionModelWithProjection = None

from transformers import CLIPImageProcessor

Image.MAX_IMAGE_PIXELS = 200_000_000
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 需要排除的模型名字（统一小写）
EXCLUDED_MODELS = {"dino_reg", "dinov2_reg_enhanced"}


# ==============================================================================
# 路径 & megfile 工具
# ==============================================================================
def is_remote_path(path: str) -> bool:
    """简单判断是否是桶路径（按需再加其他 schema）。"""
    if not isinstance(path, str):
        return False
    return path.startswith(("s3://", "oss://", "cos://", "meg://"))


def join_path(root: str, name: str) -> str:
    return root.rstrip("/") + "/" + name.lstrip("/")


def dir_exists(path: str) -> bool:
    if is_remote_path(path):
        return smart_exists(path) and smart_isdir(path)
    return os.path.isdir(path)


def file_exists(path: str) -> bool:
    if is_remote_path(path):
        return smart_exists(path)
    return os.path.isfile(path)


def norm_for_json(path: str) -> str:
    """写进 JSON 的路径：本地用绝对路径，桶路径原样返回。"""
    if is_remote_path(path):
        return path
    return os.path.abspath(path)


def write_json_anywhere(path: str, data: dict):
    """本地 / 桶 通用写 JSON。"""
    if is_remote_path(path):
        parent = path.rsplit("/", 1)[0] if "/" in path else path
        smart_makedirs(parent, exist_ok=True)
        with mopen(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def read_json_anywhere(path: str) -> Optional[dict]:
    """本地 / 桶 通用读 JSON。读取失败时返回 None。"""
    try:
        if is_remote_path(path):
            with mopen(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logging.warning(f"[WARN] 读取 JSON {path} 失败: {e}")
        return None


def save_tensor_anywhere(path: str, tensor: torch.Tensor):
    """本地 / 桶 通用保存 tensor（torch.save）。"""
    if is_remote_path(path):
        parent = path.rsplit("/", 1)[0] if "/" in path else path
        smart_makedirs(parent, exist_ok=True)
        with mopen(path, "wb") as f:
            torch.save(tensor, f)
    else:
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "wb") as f:
            torch.save(tensor, f)


def load_tensor_anywhere(path: str) -> torch.Tensor:
    """本地 / 桶 通用读取 tensor（torch.load）。"""
    if is_remote_path(path):
        with mopen(path, "rb") as f:
            return torch.load(f, map_location="cpu")
    else:
        with open(path, "rb") as f:
            return torch.load(f, map_location="cpu")


def iter_model_dirs(root: str) -> List[str]:
    """
    枚举 root 下的一级子目录，兼容本地和桶。
    """
    root = str(root).rstrip("/")
    dirs: List[str] = []

    if is_remote_path(root):
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
        p = Path(root)
        if not p.is_dir():
            return []
        for sub in p.iterdir():
            if sub.is_dir():
                dirs.append(str(sub))

    dirs.sort()
    return dirs


# ============ NEW: 从 id-list 每一行解析 style_model_id ============
def parse_style_id_from_line(line: str) -> Optional[str]:
    """
    从一行字符串中解析出 style_model_id：
    支持：
        1234567
        1234567.png
        s3://.../path/1234567.png
    取 basename，再去掉扩展名。
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    base = os.path.basename(s)
    stem, _ = os.path.splitext(base)
    stem = stem.strip()
    if not stem:
        return None
    return stem


def load_id_list_anywhere(txt_path: str) -> Set[str]:
    """
    读取 txt 中的 model_id 列表，支持本地 / 桶。

    每行可以是：
        1234567
        1234567.png
        s3://.../path/1234567.png

    统一解析为 style_model_id（去掉路径与扩展名）。
    """
    ids: Set[str] = set()
    if is_remote_path(txt_path):
        with mopen(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    for line in lines:
        style_id = parse_style_id_from_line(line)
        if style_id is None:
            continue
        ids.add(style_id)
    return ids


# ==============================================================================
# 日志 & 设备
# ==============================================================================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# 通用工具：列举文件 & 打开图片
# ==============================================================================
def list_images_recursive(root: str) -> List[str]:
    """
    递归列出 root 下所有图片文件。
    - 本地：用 Path.rglob
    - 桶：用 megfile.smart_scandir 深度优先
    """
    root = str(root).rstrip("/")

    if is_remote_path(root):
        paths: List[str] = []
        stack = [root]

        while stack:
            cur = stack.pop()
            try:
                for entry in smart_scandir(cur):
                    try:
                        if entry.is_dir():
                            stack.append(entry.path)
                        else:
                            _, ext = os.path.splitext(entry.name)
                            if ext.lower() in IMG_EXTS:
                                paths.append(entry.path)
                    except Exception:
                        continue
            except FileNotFoundError:
                continue

        return sorted(paths)

    # 本地路径
    return sorted(
        str(p) for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def open_image_anywhere(path: str) -> Image.Image:
    """本地 / 桶 通用读取图片。"""
    if is_remote_path(path):
        with mopen(path, "rb") as f:
            img = Image.open(f)
            img.load()
    else:
        img = Image.open(path)
    return img.convert("RGB")


def list_triplet_images(pair_dir: str) -> List[str]:
    """
    列出一个 <style_id>__<content_id> 目录下需要计算的所有图片。

    兼容两种情况：
    1）老版三元组结构：
        <pair_dir>/
          ├─ style_100/
          ├─ content_100/
          └─ two_100/
        -> 递归统计所有 style_/content_/two_ 子目录里的图片

    2）新版“纯图片”结构：
        <pair_dir>/
          ├─ 00001.png
          ├─ 00002.png
          └─ ...
        -> 直接把 pair_dir 下的一层图片全拿来算
    """
    # 先尝试老逻辑：找 style_/content_/two_ 子目录
    subdirs = iter_model_dirs(pair_dir)
    target_dirs: List[str] = []
    for d in subdirs:
        base = os.path.basename(str(d).rstrip("/"))
        if base.startswith(("style_", "content_", "two_")):
            target_dirs.append(d)

    # 情况 1：有三元组子目录，走老逻辑
    if target_dirs:
        paths: List[str] = []
        for d in target_dirs:
            paths.extend(list_images_recursive(d))
        paths.sort()
        return paths

    # 情况 2：没有三元组子目录，把 pair_dir 当作“纯图片目录”
    logging.info(
        f"[INFO] {pair_dir}: 未发现 style_/content_/two_ 子目录，"
        f"将直接使用该目录下一层的所有图片文件"
    )
    paths: List[str] = []

    if is_remote_path(pair_dir):
        # 桶路径：只看当前这一层的文件（不递归）
        try:
            for entry in smart_scandir(pair_dir):
                try:
                    if entry.is_dir():
                        continue
                    _, ext = os.path.splitext(entry.name)
                    if ext.lower() in IMG_EXTS:
                        paths.append(entry.path)
                except Exception:
                    continue
        except FileNotFoundError:
            pass
    else:
        # 本地路径：只看当前这一层的文件（不递归）
        p = Path(pair_dir)
        if p.is_dir():
            for f in p.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    paths.append(str(f))

    paths.sort()
    if not paths:
        logging.warning(f"[WARN] {pair_dir}: 目录下未找到任何图片")
    return paths

# ==============================================================================
# 各种 encoder 后端实现
# ==============================================================================

# ---- DINOv2 + AdaIN ----
def adain_pool(tokens: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = tokens.mean(dim=1, keepdim=True)
    std = tokens.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    norm = (tokens - mu) / std
    return norm.mean(dim=1)


class Dinov2Preprocess:
    def __init__(self, model_id_or_path: str, size: int = 518):
        assert AutoImageProcessor is not None, "需要 transformers.AutoImageProcessor"
        self.processor = AutoImageProcessor.from_pretrained(model_id_or_path)
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        out = self.processor(
            images=img, return_tensors="pt",
            do_resize=True, size={"height": self.size, "width": self.size},
            do_center_crop=False
        )
        return out["pixel_values"][0]


def build_dinov2_adain(model_id_or_path: str, size: int = 518):
    assert AutoModel is not None, "需要 transformers.AutoModel"
    device = resolve_device()
    preprocess = Dinov2Preprocess(model_id_or_path, size=size)
    model = AutoModel.from_pretrained(model_id_or_path)
    model.eval().to(device)
    return model, preprocess, device


@torch.inference_mode()
def encode_dinov2_adain(model, x: torch.Tensor, device: str) -> torch.Tensor:
    out = model(pixel_values=x.to(device))
    h = getattr(out, "last_hidden_state", None)
    if h is None:
        h = out[0] if isinstance(out, (list, tuple)) else out
    tokens = h[:, 1:, :]
    vec = adain_pool(tokens)
    return torch.nn.functional.normalize(vec.float(), dim=1)


@torch.inference_mode()
def encode_dinov2_reg_enhanced(model, x: torch.Tensor, device: str, eps: float = 1e-6) -> torch.Tensor:
    out = model(pixel_values=x.to(device))
    local_feat = out.last_hidden_state[:, 6:, :]
    global_feat = out.last_hidden_state[:, 1:6, :]
    dot_dim = local_feat @ global_feat.transpose(1, 2) / (local_feat.shape[-1] ** 0.5)
    dot_dim = dot_dim.softmax(dim=-1).mean(-1).unsqueeze(-1)  # (B,N,1)
    local_feat = local_feat * dot_dim
    mu = local_feat.mean(dim=1, keepdim=True)
    std = local_feat.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    norm = (local_feat - mu) / std
    return norm.flatten(1)


# ---- CSD ----
def build_csd_model(arch: str, model_path: str):
    assert CSD_CLIP is not None, "未安装/导入 CSD 依赖"
    device = resolve_device()
    model = CSD_CLIP(arch, "default")
    if has_batchnorms and has_batchnorms(model):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = convert_state_dict(ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    preprocess = transforms_branch0
    return model, preprocess, device


@torch.inference_mode()
def encode_csd(model, x: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(x) if hasattr(model, "encode_image") else model(x)
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    if feats.ndim > 2:
        feats = feats.flatten(1)
    return torch.nn.functional.normalize(feats, dim=1)


# ---- CLIP ----
def build_clip_model(arch: str):
    device = resolve_device()
    mapping = {"vit_large": "ViT-L/14", "vit_base": "ViT-B/16"}
    assert (local_clip is not None or pip_clip is not None), "未找到 CLIP 依赖"
    if local_clip is not None:
        model, preprocess = local_clip.load(mapping.get(arch, arch), device=device)
    else:
        model, preprocess = pip_clip.load(mapping.get(arch, arch), device=device)
    model.eval()
    return model, preprocess, device


@torch.inference_mode()
def encode_clip(model, x: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(x) if hasattr(model, "encode_image") else model.visual(x)
    if feats.ndim > 2:
        feats = feats[:, 0, :]
    return torch.nn.functional.normalize(feats, dim=1)


# ---- SigLIP ----
class SiglipPreprocess:
    def __init__(self, processor: "AutoProcessor", size=384):
        self.processor = processor
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.processor(
            images=img, return_tensors="pt",
            do_resize=True, size={"height": self.size, "width": self.size},
            do_center_crop=False)["pixel_values"][0]


def build_siglip_model(siglip_id: str):
    assert AutoProcessor is not None and AutoModel is not None, "缺少 transformers 依赖"
    device = resolve_device()
    processor = AutoProcessor.from_pretrained(siglip_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(siglip_id, trust_remote_code=True)
    model.eval().to(device, dtype=torch.float32)
    preprocess = SiglipPreprocess(processor)
    return model, preprocess, device


@torch.inference_mode()
def encode_siglip(model, x: torch.Tensor, device: str) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.float32)
    if hasattr(model, "get_image_features"):
        feats = model.get_image_features(pixel_values=x)
    else:
        raise RuntimeError("该 SigLIP 模型缺少 get_image_features()")
    if feats.ndim == 3:
        feats = feats.mean(dim=1)
    return torch.nn.functional.normalize(feats, dim=1)


# ---- VGG16 ----
class TorchvisionTensorPreprocess:
    def __init__(self, size: int = 224, mean=None, std=None):
        assert transforms is not None, "缺少 torchvision.transforms"
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        self.tf = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img)


def build_vgg16_model():
    assert vgg16 is not None and create_feature_extractor is not None, "缺少 torchvision.models"
    device = resolve_device()
    base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device)
    extractor = create_feature_extractor(base, return_nodes={"classifier.5": "feat"}).eval().to(device)
    preprocess = TorchvisionTensorPreprocess(size=224)
    return extractor, preprocess, device


@torch.inference_mode()
def encode_vgg16(extractor: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    x = x.to(device)
    out = extractor(x)["feat"]  # [N,4096]
    return torch.nn.functional.normalize(out.float(), dim=1)


# ---- SSCD ----
class SSCDModel(nn.Module):
    def __init__(self, ts_model: torch.jit.ScriptModule):
        super().__init__()
        self.inner = ts_model

    def forward(self, x):
        return self.inner(x)


def build_sscd_model(arch: str, sscd_path: Optional[str] = None):
    device = resolve_device()
    default_large = "/data/Sref_Cref/CSD/pretrainedmodels/sscd_disc_mixup.torchscript.pt"
    if sscd_path is None or not os.path.isfile(sscd_path):
        sscd_path = default_large if arch == "resnet50_disc" else sscd_path
        logging.info(f"[SSCD] 使用权重: {sscd_path}")
    ts_model = torch.jit.load(sscd_path, map_location="cpu")
    model = SSCDModel(ts_model).eval().to(device)
    preprocess = TorchvisionTensorPreprocess(size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return model, preprocess, device


@torch.inference_mode()
def encode_sscd(model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.float32)
    feats = model(x)
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    return torch.nn.functional.normalize(feats.float(), dim=1)


# ---- DINO / MoCo ----
@torch.inference_mode()
def encode_vit_like(model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    x = x.to(device)
    feat = None
    if hasattr(model, "get_intermediate_layers"):
        try:
            inter = model.get_intermediate_layers(x, n=1)[0]
            feat = inter[:, 0] if inter.ndim == 3 and inter.shape[1] > 1 else inter.mean(dim=1)
        except Exception:
            feat = None
    if feat is None and hasattr(model, "forward_features"):
        out = model.forward_features(x)
        if isinstance(out, dict):
            for k in ("x_norm_clstoken", "cls_token", "last", "feat"):
                if k in out:
                    v = out[k]
                    feat = v[:, 0] if v.ndim == 3 else v
                    break
        else:
            feat = out[:, 0] if out.ndim == 3 else out
    if feat is None:
        out = model(x)
        feat = out[:, 0] if out.ndim == 3 else out
    return torch.nn.functional.normalize(feat.float(), dim=1)


def build_dino_model(arch: str):
    assert dino_vits is not None, "未找到 models.dino_vits"
    device = resolve_device()
    mapping = {"vit_base": "dino_vitb16", "vit_base8": "dino_vitb8"}
    if arch not in mapping:
        raise NotImplementedError(f"DINO 不支持 arch={arch}")
    ctor = getattr(dino_vits, mapping[arch])
    model = ctor(pretrained=True).eval().to(device)
    preprocess = TorchvisionTensorPreprocess(size=224)
    return model, preprocess, device


def build_moco_model(arch: str, ckpt_path: Optional[str] = None):
    assert moco_vits is not None, "未找到 models.moco_vits"
    device = resolve_device()
    if arch != "vit_base":
        raise NotImplementedError("MoCo 仅支持 vit_base")
    model = getattr(moco_vits, arch)()
    if ckpt_path is None:
        ckpt_path = "/data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    new_sd = {k[len("module.base_encoder."):]: v for k, v in state_dict.items()
              if k.startswith("module.base_encoder.")}
    if not new_sd:
        new_sd = state_dict
    model.load_state_dict(new_sd, strict=False)
    model.eval().to(device)
    preprocess = TorchvisionTensorPreprocess(size=224)
    return model, preprocess, device


# ---- FLUX VAE ----
def build_flux_vae_model(ckpt_path: str, size: int = 1024):
    from diffusers import AutoencoderKL
    device = resolve_device()
    model = AutoencoderKL.from_pretrained(ckpt_path).to(device).eval()
    preprocess = TorchvisionTensorPreprocess(size=size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return model, preprocess, device


# ---- StyleShot ----
class StyleShotPreprocess:
    def __init__(self, side: int = 512):
        assert transforms is not None, "需要 torchvision.transforms"
        self.crop = transforms.Compose([
            transforms.Resize(side, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(side),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @staticmethod
    def _crop4(image: Image.Image):
        w = image.size[0]
        s = w // 2
        return (
            image.crop((0, 0, s, s)),
            image.crop((0, s, s, 2 * s)),
            image.crop((s, 0, 2 * s, s)),
            image.crop((s, s, 2 * s, 2 * s)),
        )

    def _hierarchical_patches(self, pil: Image.Image):
        high, mid, low = [], [], []
        c4 = self._crop4(pil)
        for c in c4:
            c8 = self._crop4(c)
            high.append(self.to_tensor(c8[0]))
            high.append(self.to_tensor(c8[3]))
            for c8_sel in (c8[1], c8[2]):
                c16 = self._crop4(c8_sel)
                mid.append(self.to_tensor(c16[0]))
                mid.append(self.to_tensor(c16[3]))
                for c16_sel in (c16[1], c16[2]):
                    c32 = self._crop4(c16_sel)
                    low.append(self.to_tensor(c32[0]))
                    low.append(self.to_tensor(c32[3]))
        idx = torch.randperm(len(high))
        high = torch.stack(high)[idx]
        idx = torch.randperm(len(mid))
        mid = torch.stack(mid)[idx]
        idx = torch.randperm(len(low))
        low = torch.stack(low)[idx]
        return {"high": high, "mid": mid, "low": low}

    def __call__(self, img: Image.Image):
        pil = self.crop(img.convert("RGB"))
        return self._hierarchical_patches(pil)


def build_styleshot_encoder(clip_path: str, weight_path: str):
    assert Style_Aware_Encoder is not None and CLIPVisionModelWithProjection is not None, \
        "缺少 ip_adapter/transformers 依赖（StyleShot）"
    device = resolve_device()
    backbone = CLIPVisionModelWithProjection.from_pretrained(clip_path)
    model = Style_Aware_Encoder(backbone).to(device, dtype=torch.float32).eval()
    sd = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    preprocess = StyleShotPreprocess(side=512)
    return model, preprocess, device


@torch.inference_mode()
def encode_styleshot(model, batch_dict: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    high, mid, low = batch_dict["high"], batch_dict["mid"], batch_dict["low"]
    B = high.shape[0]
    outs = []
    for i in range(B):
        tpl = (
            high[i].to(device, dtype=torch.float32, non_blocking=True),
            mid[i].to(device, dtype=torch.float32, non_blocking=True),
            low[i].to(device, dtype=torch.float32, non_blocking=True),
        )
        vec = model(tpl)
        if isinstance(vec, (list, tuple)):
            vec = vec[0]
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        vec = vec.flatten(1)
        outs.append(vec)
    feat = torch.cat(outs, dim=0)
    return torch.nn.functional.normalize(feat.float(), dim=1)


# ---- OneIG ----
class OneIGPreprocess:
    def __init__(self, model_id_or_path: str):
        self.processor = CLIPImageProcessor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        out = self.processor(images=img.convert("RGB"), return_tensors="pt")
        return out["pixel_values"][0]


def build_oneig_encoder(oneig_path: str):
    assert AutoModel is not None, "需要 transformers.AutoModel"
    device = resolve_device()
    preprocess = OneIGPreprocess(oneig_path)
    model = AutoModel.from_pretrained(oneig_path)
    model.eval().to(device)
    return model, preprocess, device


@torch.inference_mode()
def encode_oneig(model, x: torch.Tensor, device: str) -> torch.Tensor:
    out = model(pixel_values=x.to(device))
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output
    elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        h = out.last_hidden_state
        feat = h[:, 0] if h.ndim == 3 else h
    else:
        h = out[0] if isinstance(out, (list, tuple)) else out
        feat = h[:, 0] if (isinstance(h, torch.Tensor) and h.ndim == 3) else h
    return torch.nn.functional.normalize(feat.float(), dim=1)


# ==============================================================================
# 通用：构建 backend + 抽特征
# ==============================================================================
def build_backend(pt: str, args):
    pt = pt.lower()
    if pt == "csd":
        assert args.model_path, "--model_path is required for csd"
        model, preprocess, device = build_csd_model(args.arch, args.model_path)
        encode = lambda x: encode_csd(model, x)
        need_fp32 = True
        is_dict_input = False
    elif pt == "clip":
        model, preprocess, device = build_clip_model(args.arch)
        encode = lambda x: encode_clip(model, x)
        need_fp32 = False
        is_dict_input = False
    elif pt == "siglip":
        model, preprocess, device = build_siglip_model(args.siglip_id)
        encode = lambda x: encode_siglip(model, x, device)
        need_fp32 = True
        is_dict_input = False
    elif pt == "vgg":
        model, preprocess, device = build_vgg16_model()
        encode = lambda x: encode_vgg16(model, x, device)
        need_fp32 = True
        is_dict_input = False
    elif pt == "sscd":
        model, preprocess, device = build_sscd_model(args.arch, args.sscd_path)
        encode = lambda x: encode_sscd(model, x, device)
        need_fp32 = True
        is_dict_input = False
    elif pt == "dino":
        model, preprocess, device = build_dino_model(args.arch)
        encode = lambda x: encode_vit_like(model, x, device)
        need_fp32 = True
        is_dict_input = False
    elif pt == "moco":
        model, preprocess, device = build_moco_model(args.arch, args.moco_ckpt)
        encode = lambda x: encode_vit_like(model, x, device)
        need_fp32 = True
        is_dict_input = False
    elif pt == "vae":
        assert args.model_path, "--model_path is required for vae"
        model, preprocess, device = build_flux_vae_model(args.model_path, size=1024)
        encode = lambda x: model.encode(x.to(device=device, dtype=torch.float32)).latent_dist.mean.flatten(1)
        need_fp32 = True
        is_dict_input = False
    elif pt == "dinov2_adain":
        model, preprocess, device = build_dinov2_adain(args.dinov2_id, size=args.dinov2_size)
        encode = lambda x: encode_dinov2_adain(model, x, device)
        need_fp32 = True
        is_dict_input = False
    elif pt == "dinov2_reg_enhanced":
        model, preprocess, device = build_dinov2_adain(args.dinov2_id, size=args.dinov2_size)
        encode = lambda x: encode_dinov2_reg_enhanced(model, x, device)
        need_fp32 = True
        is_dict_input = False
    elif pt == "styleshot":
        model, preprocess, device = build_styleshot_encoder(
            args.styleshot_clip_path, args.styleshot_weight_path
        )
        encode = lambda batch: encode_styleshot(model, batch, device)
        need_fp32 = True
        is_dict_input = True
    elif pt == "oneig":
        model, preprocess, device = build_oneig_encoder(args.oneig_path)
        encode = lambda x: encode_oneig(model, x, device)
        need_fp32 = True
        is_dict_input = False
    else:
        raise NotImplementedError(f"unknown backend: {pt}")
    return preprocess, encode, need_fp32, is_dict_input


def load_and_embed(
    img_path: str,
    preprocess,
    encode,
    need_fp32: bool,
    is_dict_input: bool,
    device: str,
) -> torch.Tensor:
    img = open_image_anywhere(img_path)
    x = preprocess(img)
    if is_dict_input:
        batch = {k: v.unsqueeze(0).to(device=device, dtype=torch.float32)
                 for k, v in x.items()}
        feat = encode(batch)
    else:
        x = x.unsqueeze(0)
        x = x.to(device=device, dtype=torch.float32) if need_fp32 else x.to(device=device)
        feat = encode(x)
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    if feat.ndim > 2:
        feat = feat.flatten(1)
    return feat.squeeze(0).detach().cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float((a * b).sum().item())


# ==============================================================================
# encoder 权重解析
# ==============================================================================
def parse_weights_arg(weights_arg: Optional[str], models: List[str]) -> Dict[str, float]:
    """
    支持两种格式：
      1) 与 models 顺序一一对应的逗号分隔 floats: "0.5,0.5,1.0"
      2) "name:weight,name2:weight2"（name 小写）
    返回: {model_name: normalized_weight}
    """
    if not weights_arg:
        if not models:
            return {}
        w = 1.0 / len(models)
        return {m: w for m in models}

    weights_arg = weights_arg.strip()

    # name:weight 形式
    if ":" in weights_arg:
        pairs = [p.strip() for p in weights_arg.split(",") if p.strip()]
        d: Dict[str, float] = {}
        for pair in pairs:
            if ":" not in pair:
                continue
            name, val = pair.split(":", 1)
            name = name.strip().lower()
            try:
                fv = float(val)
            except Exception:
                logging.warning(f"无法解析权重 {pair}, 忽略")
                continue
            d[name] = fv

        unspecified = [m for m in models if m not in d]
        if unspecified:
            rem = max(0.0, 1.0 - sum(d.values()))
            if rem <= 0:
                for m in unspecified:
                    d[m] = 0.0
            else:
                per = rem / len(unspecified)
                for m in unspecified:
                    d[m] = per

        s = sum(d.values())
        if s <= 0:
            per = 1.0 / max(1, len(models))
            return {m: per for m in models}
        return {m: (d.get(m, 0.0) / s) for m in models}

    # comma-separated floats 与 models 顺序一一对应
    parts = [p.strip() for p in weights_arg.split(",") if p.strip()]
    try:
        floats = [float(p) for p in parts]
    except Exception:
        logging.warning("解析权重列表失败，使用均匀权重")
        per = 1.0 / max(1, len(models))
        return {m: per for m in models}

    if len(floats) != len(models):
        logging.warning("权重数量与模型数量不一致，使用均匀权重")
        per = 1.0 / max(1, len(models))
        return {m: per for m in models}

    s = sum(floats)
    if s == 0:
        per = 1.0 / max(1, len(models))
        return {m: per for m in models}

    return {models[i]: floats[i] / s for i in range(len(models))}


# ==============================================================================
# 单张图片特征提取（用于任意图片）
# ==============================================================================
def compute_feats_for_image(
    img_path: str,
    backends: Dict[str, Tuple],
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    为单张图片在所有 backends 上提特征并 L2 归一化。
    返回 {model_name: feat (cpu,1-D)}。
    """
    feats: Dict[str, torch.Tensor] = {}
    for model_name, (preprocess, encode, need_fp32, is_dict_input) in list(backends.items()):
        try:
            feat = load_and_embed(img_path, preprocess, encode, need_fp32, is_dict_input, device)
        except Exception as e:
            logging.error(f"[{model_name}] 提取 {img_path} 特征失败: {e}")
            continue
        feat = feat.detach().cpu().float()
        feat = feat / (feat.norm(p=2) + 1e-8)
        feats[model_name] = feat
    return feats


# ==============================================================================
# 多进程部分：全局变量 + style mean 计算
# ==============================================================================
# 这些全局变量在每个 worker 进程中各自维护一份
G_BACKENDS: Dict[str, Tuple] = {}
G_DEVICE: str = "cpu"
G_WEIGHTS: Dict[str, float] = {}
G_CFG: Dict[str, object] = {}
G_STYLE_MEANS: Dict[str, Dict[str, torch.Tensor]] = {}  # style_id -> {encoder_name: mean_vec}


def find_style_dir_for_model(style_root: str, style_id: str) -> Optional[str]:
    """在 style_root/<style_id>/ 下寻找第一个以 style_ 开头的目录。"""
    model_dir = join_path(style_root, style_id)
    if not dir_exists(model_dir):
        logging.warning(f"[WARN] style_id={style_id}: 目录不存在: {model_dir}")
        return None
    subdirs = iter_model_dirs(model_dir)
    for d in subdirs:
        base = os.path.basename(str(d).rstrip("/"))
        if base.startswith("style_"):
            return d
    logging.warning(f"[WARN] style_id={style_id}: 未找到以 style_ 开头的子目录")
    return None


def ensure_style_means(style_id: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    确保当前 worker 已经有该 style_id 的 style mean：
      1) 若内存中已有，直接返回
      2) 若磁盘存在 <encoder_name>_mean.pth，读取并归一化
      3) 否则从 style_100 目录所有图片抽特征做均值，并可选写回 mean.pth
    """
    global G_STYLE_MEANS, G_BACKENDS, G_DEVICE, G_CFG

    if style_id in G_STYLE_MEANS:
        means = G_STYLE_MEANS[style_id]
        return means if means else None

    style_root = G_CFG["style_root"]  # type: ignore
    save_mean = not bool(G_CFG.get("no_save_mean", False))

    if not isinstance(style_root, str):
        logging.error("[FATAL] G_CFG['style_root'] 非字符串")
        G_STYLE_MEANS[style_id] = {}
        return None

    model_dir = join_path(style_root, style_id)
    style_dir = find_style_dir_for_model(style_root, style_id)
    if style_dir is None:
        G_STYLE_MEANS[style_id] = {}
        return None

    means: Dict[str, torch.Tensor] = {}
    # 先尝试读取已有 mean.pth
    for encoder_name in G_BACKENDS.keys():
        mean_name = f"{encoder_name}_mean.pth"
        mean_path = join_path(model_dir, mean_name)
        if file_exists(mean_path):
            try:
                t = load_tensor_anywhere(mean_path)
                t = torch.as_tensor(t, dtype=torch.float32).flatten()
                t = t / (t.norm(p=2) + 1e-8)
                means[encoder_name] = t
                logging.info(f"[style_id={style_id}] 读取已有 mean: {mean_path}")
            except Exception as e:
                logging.warning(f"[WARN] 读取 mean 失败 {mean_path}: {e}")

    missing = [name for name in G_BACKENDS.keys() if name not in means]

    if missing:
        style_imgs = list_images_recursive(style_dir)
        if not style_imgs:
            logging.warning(
                f"[WARN] style_id={style_id}: style_dir={style_dir} 下没有图片，无法计算均值"
            )
            G_STYLE_MEANS[style_id] = means
            return means if means else None

        logging.info(
            f"[style_id={style_id}] 在 {style_dir} 中找到 {len(style_imgs)} 张风格参考图，"
            f"需要为 {missing} 计算 mean"
        )

        sums: Dict[str, torch.Tensor] = {}
        cnts: Dict[str, int] = {}
        for img_path in style_imgs:
            feats = compute_feats_for_image(img_path, G_BACKENDS, G_DEVICE)
            for name in missing:
                if name not in feats:
                    continue
                feat = feats[name]
                if name not in sums:
                    sums[name] = feat.clone()
                    cnts[name] = 1
                else:
                    sums[name] += feat
                    cnts[name] += 1

        for name in missing:
            if name not in sums or cnts.get(name, 0) == 0:
                logging.warning(
                    f"[WARN] style_id={style_id}: encoder={name} 在 style_100 上无有效特征，跳过"
                )
                continue
            mean_vec = sums[name] / cnts[name]
            mean_vec = mean_vec / (mean_vec.norm(p=2) + 1e-8)
            means[name] = mean_vec
            if save_mean:
                mean_path = join_path(model_dir, f"{name}_mean.pth")
                try:
                    save_tensor_anywhere(mean_path, mean_vec.cpu())
                    logging.info(
                        f"[style_id={style_id}] 已保存 encoder={name} 的 mean 至 {mean_path}"
                    )
                except Exception as e:
                    logging.warning(
                        f"[WARN] style_id={style_id}: 保存 mean 失败 {mean_path}: {e}"
                    )

    if not means:
        logging.warning(f"[WARN] style_id={style_id}: 最终没有可用的 style mean")
        G_STYLE_MEANS[style_id] = {}
        return None

    G_STYLE_MEANS[style_id] = means
    return means


def init_worker(cfg: Dict[str, object]):
    """
    每个进程启动时调用一次：构建 backends + encoder 权重。
    支持按 gpu_ids 实现“一块 GPU 一个进程”：
      - 若 cfg["gpu_ids"] 存在，则：
          * 对于 Pool worker：根据 worker 序号绑定到 gpu_ids[worker_idx]
          * 对于单进程模式：绑定到 gpu_ids[0]
    """
    global G_BACKENDS, G_DEVICE, G_WEIGHTS, G_CFG, G_STYLE_MEANS

    setup_logging()
    G_CFG = cfg
    G_STYLE_MEANS = {}

    gpu_ids: Optional[List[int]] = cfg.get("gpu_ids")  # type: ignore
    if gpu_ids:
        proc = mp.current_process()
        if getattr(proc, "_identity", None):
            worker_idx = proc._identity[0] - 1  # 0-based
        else:
            worker_idx = 0
        gpu_id = gpu_ids[worker_idx % len(gpu_ids)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logging.info(
            f"[Worker {os.getpid()}] 绑定 GPU {gpu_id} "
            f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
        )

    G_DEVICE = resolve_device()
    logging.info(f"[Worker {os.getpid()}] 使用设备: {G_DEVICE}")

    pt_style_flat: List[str] = cfg["pt_style_flat"]  # type: ignore

    DummyArgs = type(
        "Args",
        (),
        {
            "arch": cfg["arch"],
            "model_path": cfg["model_path"],
            "siglip_id": cfg["siglip_id"],
            "sscd_path": cfg["sscd_path"],
            "moco_ckpt": cfg["moco_ckpt"],
            "dinov2_id": cfg["dinov2_id"],
            "dinov2_size": cfg["dinov2_size"],
            "styleshot_clip_path": cfg["styleshot_clip_path"],
            "styleshot_weight_path": cfg["styleshot_weight_path"],
            "oneig_path": cfg["oneig_path"],
        },
    )
    args_obj = DummyArgs()

    backends: Dict[str, Tuple] = {}
    for pt in pt_style_flat:
        try:
            preprocess, encode, need_fp32, is_dict_input = build_backend(pt, args_obj)
            backends[pt] = (preprocess, encode, need_fp32, is_dict_input)
            logging.info(f"[Worker {os.getpid()}] [OK] 后端 {pt} 构建成功")
        except Exception as e:
            logging.error(f"[Worker {os.getpid()}] [WARN] 后端 {pt} 构建失败，跳过: {e}")

    if not backends:
        raise RuntimeError(f"[Worker {os.getpid()}] 没有可用的后端模型，退出")

    G_BACKENDS = backends
    G_WEIGHTS = parse_weights_arg(cfg["encoder_weights"], list(backends.keys()))  # type: ignore
    logging.info(f"[Worker {os.getpid()}] encoder weights: {G_WEIGHTS}")


def worker_process(task):
    """
    task: (idx, pair_dir)
    对单个 <style_id>__<content_id> 目录下的所有 style_/content_/two_ 子目录中的图片，
    计算它们相对于 style_id 对应 style_100 均值风格的相似度。
    返回: {image_path(norm): weighted_score}

    同时会在该 pair_dir 下写出 triplet_style_similarity.json。
    """
    global G_BACKENDS, G_DEVICE, G_WEIGHTS, G_CFG

    idx, pair_dir = task
    pair_dir = str(pair_dir).rstrip("/")

    base = os.path.basename(pair_dir)
    if "__" not in base:
        logging.warning(f"[SKIP] {pair_dir}: 目录名不符合 <style_id>__<content_id> 格式")
        return {}

    style_id, _ = base.split("__", 1)


    style_means = ensure_style_means(style_id)
    if not style_means:
        logging.warning(f"[SKIP] {pair_dir}: style_id={style_id} 无可用 style mean，跳过该三元组目录")
        return {}

    img_paths = list_triplet_images(pair_dir)
    if not img_paths:
        logging.warning(f"[SKIP] {pair_dir}: 未找到三元组图片，跳过")
        return {}

    result: Dict[str, float] = {}
    logging.info(f"[PAIR_DIR] {pair_dir}: 将对 {len(img_paths)} 张图片计算风格相似度")

    for img_path in img_paths:
        feats = compute_feats_for_image(img_path, G_BACKENDS, G_DEVICE)
        if not feats:
            continue

        total = 0.0
        wsum = 0.0
        for encoder_name, mean_vec in style_means.items():
            feat = feats.get(encoder_name)
            if feat is None:
                continue
            sim = cosine_sim(feat, mean_vec)
            w = G_WEIGHTS.get(encoder_name, 0.0)
            total += w * sim
            wsum += w

        if wsum <= 0:
            continue

        score = total / wsum
        result[norm_for_json(img_path)] = float(score)

    logging.info(f"[DONE] {pair_dir}: 计算得到 {len(result)} 条图片风格相似度结果")

    # 每个三元组目录下写出局部 JSON：triplet_style_similarity.json
    if result:
        out_json_path = join_path(pair_dir, "triplet_style_similarity.json")
        out_data = {
            "pair_dir": norm_for_json(pair_dir),
            "style_id": style_id,
            "num_images": len(result),
            "pt_style": list(G_BACKENDS.keys()),
            "encoder_weights_used": G_WEIGHTS,
            "scores": result,
        }
        write_json_anywhere(out_json_path, out_data)
        logging.info(
            f"[PAIR_DIR] {pair_dir}: 已写入 triplet_style_similarity.json ({len(result)} 条记录)"
        )

    return result


# ==============================================================================
# CLI：遍历 root，多进程处理
# ==============================================================================
def main():
    setup_logging()
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "遍历 --root 下的 <style_id>__<content_id> 子目录，"
            "对其中 style_/content_/two_ 下的所有图片，"
            "计算它们相对于对应 style_id 的 style_100 均值风格相似度，"
            "结果汇总写入单个 JSON（image_path -> score，支持多 encoder 加权、多进程 & 桶路径），"
            "同时在每个三元组子目录下写出 triplet_style_similarity.json。"
        )
    )

    ap.add_argument(
        "--root",
        required=True,
        help="包含多个 <style_id>__<content_id> 子目录的根目录（本地或 s3:// 等桶路径），即三元组目录",
    )
    ap.add_argument(
        "--style-root",
        required=True,
        help="包含各个 <model_id> 子目录的根目录，用于在 <model_id>/style_100 下计算 style mean",
    )
    ap.add_argument(
        "--output-json",
        required=True,
        help="最终汇总输出的 JSON 路径（本地或桶），结构中包含 scores: {image_path: score}",
    )
    ap.add_argument(
        "--id-list",
        default=None,
        help=(
            "可选：txt 文件；每行可以是 model_id、model_id.png，或带路径的 .../model_id.png；"
            "脚本会从 basename 去掉扩展名后得到 style_model_id，"
            "仅处理这些 style_model_id 对应的 <style_id>__<content_id> 目录；"
            "不填写则处理所有三元组目录。"
        ),
    )
    ap.add_argument(
        "--no-save-mean",
        action="store_true",
        help="若指定，则只在内存中缓存 style mean，不写回 <encoder_name>_mean.pth；"
             "默认会写回以便下次复用。",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（目前主要用于 StyleShot 的 patch shuffle 等）",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="进程数；=1 表示单进程顺序跑；>=2 时使用多进程。",
    )
    ap.add_argument(
        "--gpu-ids",
        default=None,
        help="指定要使用的 GPU id 列表，例如 '0,1,2,3'，会按“一块 GPU 一个进程”方式绑定；"
             "若不指定，则按默认的 resolve_device() 逻辑运行。",
    )

    # 下面这些参数直接转发给 encoder 构建
    ap.add_argument(
        "--pt-style",
        action="append",
        default=[],
        help="可多次传入或用逗号分隔：csd,clip,siglip,vgg,sscd,dino,moco,vae,"
             "dinov2_adain,styleshot,oneig,dinov2_reg_enhanced（会自动排除 dino_reg / dinov2_reg_enhanced）",
    )
    ap.add_argument("--arch", default="vit_base")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--siglip_id",
                    default=os.getenv("SIGLIP_PATH", "google/siglip-so400m-patch14-384"))
    ap.add_argument("--sscd_path", default=None)
    ap.add_argument("--moco_ckpt", default=None)
    ap.add_argument("--dinov2_id",
                    default=os.getenv("DINOV2_PATH", "facebook/dinov2-base"))
    ap.add_argument("--dinov2_size", type=int, default=518)
    ap.add_argument("--styleshot_clip_path",
                    default=os.getenv(
                        "STYLESHOT_CLIP",
                        "/mnt/jfs/model_zoo/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    ))
    ap.add_argument("--styleshot_weight_path",
                    default=os.getenv(
                        "STYLESHOT_WEIGHT",
                        "/mnt/jfs/model_zoo/StyleShot/StyleShot/pretrained_weight/style_aware_encoder.bin",
                    ))
    ap.add_argument("--oneig_path",
                    default=os.getenv("ONEIG_PATH", "xingpng/OneIG-StyleEncoder"))
    ap.add_argument(
        "--encoder-weights",
        default=None,
        help="encoder 权重；两种格式：\n"
             "  1) 与 --pt-style 展开顺序对应的逗号浮点，如 '0.5,0.3,0.2'；\n"
             "  2) 'name:weight,name2:weight2'（name 小写）",
        )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "如指定，则无视已存在的 <style_id>__<content_id>/triplet_style_similarity.json，"
            "重新计算并覆盖；"
            "默认情况下若该文件已存在，则跳过该 pair 的重新计算，并复用已有 scores。"
        ),
    )

    args = ap.parse_args()
    pair_root = args.root.rstrip("/")
    style_root = args.style_root.rstrip("/")
    random.seed(args.seed)

    # 解析 gpu_ids
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [
            int(x) for x in re.split(r"[,\s]+", args.gpu_ids.strip())
            if x != ""
        ]
        if not gpu_ids:
            gpu_ids = None

    # 一块 GPU 一个进程：如果给了 gpu_ids 且 num_workers 还是默认 1，就自动改成 GPU 数量
    num_workers = args.num_workers
    if gpu_ids is not None:
        if num_workers == 1:
            num_workers = len(gpu_ids)
        elif num_workers != len(gpu_ids):
            logging.warning(
                f"[WARN] --num-workers={num_workers} 和 --gpu-ids 数量 {len(gpu_ids)} 不一致，"
                f"仍会创建 {num_workers} 个进程，GPU 会按列表循环复用；"
                f"如果想一块 GPU 一个进程，建议 num-workers == len(gpu-ids)。"
            )

    model_dirs = iter_model_dirs(pair_root)
    if not model_dirs:
        raise SystemExit(f"在 root={pair_root} 下没有找到任何子目录")

    # 若给了 id-list，则只保留 style_id 在列表中的 pair 目录
    allowed_ids: Optional[Set[str]] = None
    if args.id_list:
        allowed_ids = load_id_list_anywhere(args.id_list)
        logging.info(f"[INFO] 从 {args.id_list} 解析到 {len(allowed_ids)} 个 style_model_id")

    filtered_dirs: List[str] = []
    for d in model_dirs:
        base = os.path.basename(str(d).rstrip("/"))
        if "__" not in base:
            continue
        style_id, _ = base.split("__", 1)
        if allowed_ids is not None and style_id not in allowed_ids:
            continue
        filtered_dirs.append(d)

    if not filtered_dirs:
        raise SystemExit("没有任何符合条件的 <style_id>__<content_id> 目录可供处理，请检查 --id-list 或目录结构")

    logging.info(
        f"[INFO] 在 {pair_root} 下找到 {len(model_dirs)} 个子目录，"
        f"其中 {len(filtered_dirs)} 个将被用于三元组风格相似度计算"
    )

    # 展开 pt-style 里的逗号写法
    pt_style_flat: List[str] = []
    for it in args.pt_style:
        if not it:
            continue
        parts = [p.strip().lower() for p in it.split(",") if p.strip()]
        pt_style_flat.extend(parts)

    # 自动排除无效模型
    if any(m in pt_style_flat for m in EXCLUDED_MODELS):
        logging.info(f"自动移除模型: {EXCLUDED_MODELS & set(pt_style_flat)}")
        pt_style_flat = [m for m in pt_style_flat if m not in EXCLUDED_MODELS]

    if not pt_style_flat:
        raise SystemExit("请至少指定一个 --pt-style（排除 dino_reg / dinov2_reg_enhanced 后为空）")

    # 预先算一份 encoder 权重字典（用于输出）
    encoder_weights_dict = parse_weights_arg(args.encoder_weights, pt_style_flat)

    # 组装给 worker 的配置（只包含可序列化的简单类型）
    cfg: Dict[str, object] = {
        "pt_style_flat": pt_style_flat,
        "arch": args.arch,
        "model_path": args.model_path,
        "siglip_id": args.siglip_id,
        "sscd_path": args.sscd_path,
        "moco_ckpt": args.moco_ckpt,
        "dinov2_id": args.dinov2_id,
        "dinov2_size": args.dinov2_size,
        "styleshot_clip_path": args.styleshot_clip_path,
        "styleshot_weight_path": args.styleshot_weight_path,
        "oneig_path": args.oneig_path,
        "encoder_weights": args.encoder_weights,
        "seed": args.seed,
        "gpu_ids": gpu_ids,
        "style_root": style_root,
        "no_save_mean": bool(args.no_save_mean),
        "overwrite": bool(args.overwrite),   # 👈 新增
    }

    tasks = [(idx, model_dir) for idx, model_dir in enumerate(filtered_dirs)]

    all_scores: Dict[str, float] = {}

    if num_workers <= 1:
        logging.info("[INFO] 使用单进程模式")
        init_worker(cfg)
        for t in tasks:
            part = worker_process(t)
            if isinstance(part, dict):
                all_scores.update(part)
    else:
        num_workers = max(1, num_workers)
        logging.info(f"[INFO] 使用多进程模式: num_workers={num_workers}")

        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(cfg,),
        ) as pool:
            for part in pool.imap_unordered(worker_process, tasks):
                if isinstance(part, dict):
                    all_scores.update(part)

    logging.info(f"[INFO] 共汇总 {len(all_scores)} 条图片风格相似度结果，将写入 {args.output_json}")

    out_data = {
        "pair_root": norm_for_json(pair_root),
        "style_root": norm_for_json(style_root),
        "pt_style": pt_style_flat,
        "encoder_weights_used": encoder_weights_dict,
        "scores": all_scores,
    }
    write_json_anywhere(args.output_json, out_data)
    logging.info("[DONE] 全部完成")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # 可能已经在其他地方设置过
        pass
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多进程版 Gallery mean vs probe / style_100 内均值 风格相似度工具（多后端，支持桶路径）。

模式一（默认）
=============
- 与原脚本一致：
  - 对每个 <root>/<model_id>/：
      demo_images/
      eval_images/
  - 从 demo_images 随机选 1 张作为 probe；
  - gallery = eval_images 全部图片；
  - 计算各 encoder 的 gallery mean 与 probe 之间的相似度；
  - 输出到 model_dir/style_similarity.json。

模式二（--style-mean）
======================
- 对每个 <root>/<model_id>/style_100（或 --style-dir-name 指定的目录）：
  - 为文件夹内所有图片提特征，先算「有效图片」的均值特征；
  - 再计算「每张图片 vs 均值」的相似度（per_model + weighted_score）；
  - 如果图片完全损坏（所有 encoder 都提特征失败）：
      * 在算均值时跳过这张图；
      * 该图的相似度记为 0（per_model similarity=0, weighted_score=0）；
  - 输出 JSON 顶层仍有：
      * weighted_score：所有有效图片 weighted_score 的平均；
      * per_model：各 encoder 的平均相似度；
    另外新增：
      * images：每张图片的详细相似度；
      * summary：统计信息（num_images / num_valid / num_corrupted 等）。
"""

import os
import re
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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


# ==============================================================================
# 各种 encoder 后端实现（和你原脚本保持一致）
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
        # 会被 EXCLUDED_MODELS 排除，正常不走到这里
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
# Gallery mean & probe 提特征（原有逻辑）
# ==============================================================================
def compute_gallery_means(
    gallery_paths: List[str],
    backends: Dict[str, Tuple],
    device: str,
):
    """
    gallery_paths: list of image paths
    backends: {model_name: (preprocess, encode, need_fp32, is_dict_input)}

    返回:
      means : {model_name: mean_feat (cpu, 1-D, 已归一化)}
      counts: {model_name: 有效图片数量}
    """
    feat_sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for idx, p in enumerate(gallery_paths):
        if idx % 50 == 0:
            logging.info(f"[Gallery] processed {idx}/{len(gallery_paths)}")
        for model_name, (preprocess, encode, need_fp32, is_dict_input) in list(backends.items()):
            try:
                feat = load_and_embed(p, preprocess, encode, need_fp32, is_dict_input, device)
            except Exception as e:
                logging.error(f"[{model_name}] 提取 {p} 特征失败: {e}")
                continue

            feat = feat.detach().cpu().float()
            if model_name not in feat_sums:
                feat_sums[model_name] = feat.clone()
                counts[model_name] = 1
            else:
                feat_sums[model_name] += feat
                counts[model_name] += 1

    means: Dict[str, torch.Tensor] = {}
    for model_name, s in feat_sums.items():
        cnt = counts.get(model_name, 1)
        mean = s / float(cnt)
        mean = mean / (mean.norm(p=2) + 1e-8)
        means[model_name] = mean
        logging.info(f"[{model_name}] gallery count={cnt}; feat_dim={mean.numel()}")

    return means, counts


def compute_probe_feats(
    probe_path: str,
    backends: Dict[str, Tuple],
    device: str,
):
    """
    返回:
      {model_name: feat (cpu, 1-D, 已归一化)}
    """
    probe_feats: Dict[str, torch.Tensor] = {}

    for model_name, (preprocess, encode, need_fp32, is_dict_input) in list(backends.items()):
        try:
            feat = load_and_embed(probe_path, preprocess, encode, need_fp32, is_dict_input, device)
        except Exception as e:
            logging.error(f"[{model_name}] 提取 probe 特征失败: {e}")
            continue

        feat = feat.detach().cpu().float()
        feat = feat / (feat.norm(p=2) + 1e-8)

        probe_feats[model_name] = feat
        logging.info(f"[{model_name}] probe feat dim={feat.numel()}")

    return probe_feats


def compute_gallery_probe_with_backends(
    gallery_dir: str,
    probe_image: str,
    output_json: str,
    backends: Dict[str, Tuple],
    weights: Dict[str, float],
    device: str,
):
    """
    使用“已构建好的 backends + 权重”计算某个 (gallery_dir, probe_image) 的风格相似度。
    出现问题时不抛异常，返回 None，交给上层决定是否跳过。
    """

    gallery_paths = list_images_recursive(gallery_dir)
    if not gallery_paths:
        logging.warning(f"[SKIP] gallery_dir({gallery_dir}) 中没有有效图片，跳过")
        return None

    if not file_exists(probe_image):
        logging.warning(f"[SKIP] probe_image ({probe_image}) 不存在，跳过")
        return None

    logging.info(f"[Gallery] dir={gallery_dir}, images={len(gallery_paths)}")
    logging.info(f"[Probe  ] image={probe_image}")

    # 计算 gallery mean
    logging.info("开始对 gallery 计算各 encoder 的 mean 特征...")
    gallery_means, counts = compute_gallery_means(gallery_paths, backends.copy(), device)
    if not gallery_means:
        logging.warning(f"[SKIP] {gallery_dir}: 未能为任何 encoder 计算 gallery mean，跳过该目录")
        return None

    # probe 特征（只为 gallery 成功的 encoder 再跑一次）
    logging.info("提取 probe 特征...")
    active_models = list(gallery_means.keys())
    backends_probe = {m: backends[m] for m in active_models if m in backends}
    probe_feats = compute_probe_feats(probe_image, backends_probe.copy(), device)

    if not probe_feats:
        logging.warning(f"[SKIP] {gallery_dir}: probe 未能计算出任何 encoder 特征，跳过该目录")
        return None

    # per-model 相似度
    per_model: Dict[str, Dict[str, float]] = {}
    for model_name, mean_feat in gallery_means.items():
        if model_name not in probe_feats:
            logging.warning(f"[{model_name}] probe 未提取到特征，跳过该 encoder")
            continue

        sim = cosine_sim(mean_feat, probe_feats[model_name])
        per_model[model_name] = {
            "similarity": float(sim),
            "gallery_count": int(counts.get(model_name, 0)),
        }

    if not per_model:
        logging.warning(f"[SKIP] {gallery_dir}: 所有 encoder 相似度均缺失，跳过该目录")
        return None

    # 加权平均
    total = 0.0
    weight_sum = 0.0
    for model_name, info in per_model.items():
        w = weights.get(model_name, 0.0)
        total += w * info["similarity"]
        weight_sum += w

    weighted_score = total / weight_sum if weight_sum > 0 else None

    output = {
        "probe_image": norm_for_json(probe_image),
        "gallery_dir": norm_for_json(gallery_dir),
        "per_model": per_model,
        "encoder_weights_used": weights,
        "weighted_score": None if weighted_score is None else float(weighted_score),
    }

    write_json_anywhere(output_json, output)
    logging.info(f"[DONE] 输出 JSON: {output_json}")

    return output


# ==============================================================================
# 新增：style_100 文件夹内「每张图 vs 均值」相似度
# ==============================================================================
def compute_style_folder_mean_and_per_image(
    style_dir: str,
    output_json: str,
    backends: Dict[str, Tuple],
    weights: Dict[str, float],
    device: str,
):
    """
    对 style_dir 下所有图片：
      1) 先为每张图提特征，累计各 encoder 的 feat_sums / counts（损坏图跳过，不进均值）；
      2) 用有效图片算出每个 encoder 的均值特征；
      3) 再对每张图计算「与均值」的相似度：
         - 正常图：按各 encoder 计算 similarity，并做 weighted_score；
         - 完全损坏的图（所有 encoder 都失败）：per_model similarity 全为 0，weighted_score=0；
    输出 JSON 顶层包含：
      - style_dir
      - encoder_weights_used
      - per_model: 各 encoder 的平均相似度（只统计有效图片）
      - weighted_score: 所有有效图片 weighted_score 的平均
      - images: 每张图片的详细信息
      - summary: num_images / num_valid / num_corrupted 等统计
    """
    img_paths = list_images_recursive(style_dir)
    if not img_paths:
        logging.warning(f"[SKIP] style_dir({style_dir}) 中没有有效图片，跳过")
        return None

    logging.info(f"[STYLE] dir={style_dir}, images={len(img_paths)}")

    # 用文件名作为 key（如果你担心重名，也可以改成相对路径）
    feat_sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    per_image_feats: Dict[str, Dict[str, torch.Tensor]] = {}
    images_info: Dict[str, Dict[str, object]] = {}
    num_corrupted = 0

    for idx, p in enumerate(img_paths):
        if idx % 50 == 0:
            logging.info(f"[STYLE] processed {idx}/{len(img_paths)}")
        name = os.path.basename(p)
        feats_for_image: Dict[str, torch.Tensor] = {}

        for model_name, (preprocess, encode, need_fp32, is_dict_input) in list(backends.items()):
            try:
                feat = load_and_embed(p, preprocess, encode, need_fp32, is_dict_input, device)
            except Exception as e:
                logging.error(f"[STYLE][{model_name}] 提取 {p} 特征失败: {e}")
                continue

            feat = feat.detach().cpu().float()
            feat = feat / (feat.norm(p=2) + 1e-8)

            feats_for_image[model_name] = feat

            if model_name not in feat_sums:
                feat_sums[model_name] = feat.clone()
                counts[model_name] = 1
            else:
                feat_sums[model_name] += feat
                counts[model_name] += 1

        if feats_for_image:
            per_image_feats[name] = feats_for_image
            images_info[name] = {
                "image": norm_for_json(p),
                "corrupted": False,
            }
        else:
            # 该图在所有 encoder 上都失败，视为损坏图片
            per_image_feats[name] = {}
            images_info[name] = {
                "image": norm_for_json(p),
                "corrupted": True,
            }
            num_corrupted += 1

    if not feat_sums:
        logging.warning(f"[SKIP] {style_dir}: 所有图片均无法用于计算均值，跳过该目录")
        return None

    # 计算各 encoder 的均值特征（仅使用成功的图片）
    means: Dict[str, torch.Tensor] = {}
    for model_name, s in feat_sums.items():
        cnt = counts.get(model_name, 1)
        mean = s / float(cnt)
        mean = mean / (mean.norm(p=2) + 1e-8)
        means[model_name] = mean
        logging.info(f"[STYLE][{model_name}] valid_count={cnt}; feat_dim={mean.numel()}")

    # 逐图片计算与均值的相似度
    images_out: Dict[str, Dict[str, object]] = {}
    weighted_scores_valid: List[float] = []
    pm_sum: Dict[str, float] = {}
    pm_cnt: Dict[str, int] = {}

    for name, base_info in images_info.items():
        img_path = base_info["image"]
        corrupted = bool(base_info.get("corrupted", False))
        feats = per_image_feats.get(name, {})
        per_model = {}

        if corrupted or not feats:
            # 损坏图：相似度为 0，均值时已跳过
            for model_name in means.keys():
                per_model[model_name] = {"similarity": 0.0}
            weighted_score = 0.0
        else:
            total = 0.0
            wsum = 0.0
            for model_name, mean_feat in means.items():
                if model_name not in feats:
                    continue
                sim = cosine_sim(mean_feat, feats[model_name])
                per_model[model_name] = {"similarity": float(sim)}

                w = weights.get(model_name, 0.0)
                total += w * sim
                wsum += w

                pm_sum[model_name] = pm_sum.get(model_name, 0.0) + sim
                pm_cnt[model_name] = pm_cnt.get(model_name, 0) + 1

            weighted_score = total / wsum if wsum > 0 else 0.0
            weighted_scores_valid.append(weighted_score)

        images_out[name] = {
            "image": img_path,
            "corrupted": corrupted,
            "per_model": per_model,
            "weighted_score": float(weighted_score),
        }

    num_images = len(images_info)
    num_valid = num_images - num_corrupted

    per_model_mean_similarity = {
        m: pm_sum[m] / pm_cnt[m]
        for m in pm_sum
        if pm_cnt[m] > 0
    }
    mean_weighted_score = (
        sum(weighted_scores_valid) / len(weighted_scores_valid)
        if weighted_scores_valid else None
    )

    output = {
        "style_dir": norm_for_json(style_dir),
        "encoder_weights_used": weights,
        # 顶层 per_model / weighted_score 保持兼容含义（这里是对「有效图片」的平均）
        "per_model": per_model_mean_similarity,
        "weighted_score": None if mean_weighted_score is None else float(mean_weighted_score),
        "images": images_out,
        "summary": {
            "num_images": int(num_images),
            "num_valid": int(num_valid),
            "num_corrupted": int(num_corrupted),
            "mean_weighted_score": None if mean_weighted_score is None else float(mean_weighted_score),
            "per_model_mean_similarity": per_model_mean_similarity,
        },
    }

    write_json_anywhere(output_json, output)
    logging.info(
        f"[DONE][STYLE] {style_dir}: 写出 JSON {output_json}，"
        f"num_images={num_images}, num_valid={num_valid}, num_corrupted={num_corrupted}"
    )
    return output


# ==============================================================================
# 多进程部分：全局变量 + worker
# ==============================================================================
# 这些全局变量在每个 worker 进程中各自维护一份
G_BACKENDS: Dict[str, Tuple] = {}
G_DEVICE: str = "cpu"
G_WEIGHTS: Dict[str, float] = {}
G_CFG: Dict[str, object] = {}


def init_worker(cfg: Dict[str, object]):
    """
    每个进程启动时调用一次：构建 backends + encoder 权重。
    """
    global G_BACKENDS, G_DEVICE, G_WEIGHTS, G_CFG

    setup_logging()
    G_CFG = cfg
    G_DEVICE = resolve_device()
    logging.info(f"[Worker {os.getpid()}] 使用设备: {G_DEVICE}")

    pt_style_flat: List[str] = cfg["pt_style_flat"]  # 已经是扁平化后的 list[str]

    # 构造 DummyArgs 给 build_backend 用
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
    G_WEIGHTS = parse_weights_arg(cfg["encoder_weights"], list(backends.keys()))
    logging.info(f"[Worker {os.getpid()}] encoder weights: {G_WEIGHTS}")


def worker_process(task):
    """
    task: (idx, model_dir)
    """
    global G_BACKENDS, G_DEVICE, G_WEIGHTS, G_CFG
    idx, model_dir = task

    output_name: str = G_CFG["output_name"]
    overwrite: bool = G_CFG["overwrite"]
    base_seed: int = G_CFG["seed"]
    style_mean_mode: bool = G_CFG.get("style_mean", False)
    style_dir_name: str = G_CFG.get("style_dir_name", "style_100")

    # ========================
    # 模式二：style_100 内均值
    # ========================
    if style_mean_mode:
        style_dir = join_path(model_dir, style_dir_name)
        out_json = join_path(model_dir, output_name)

        if not dir_exists(style_dir):
            logging.info(f"[SKIP] {model_dir}: 未找到 style 目录 {style_dir}，跳过")
            return None

        if (not overwrite) and smart_exists(out_json):
            logging.info(f"[SKIP] {model_dir}: {out_json} 已存在（未指定 --overwrite），跳过")
            return None

        try:
            result = compute_style_folder_mean_and_per_image(
                style_dir=style_dir,
                output_json=out_json,
                backends=G_BACKENDS,
                weights=G_WEIGHTS,
                device=G_DEVICE,
            )
            if result is None:
                return None
        except Exception as e:
            logging.error(f"[ERROR][STYLE] {model_dir}: 计算 style 均值相似度失败 ----> {e}")
            return None

        return out_json

    # ========================
    # 模式一：原始 demo / eval gallery-probe
    # ========================
    demo_dir = join_path(model_dir, "demo_images")
    eval_dir = join_path(model_dir, "eval_images")
    out_json = join_path(model_dir, output_name)

    # 条件检查
    if not dir_exists(demo_dir) or not dir_exists(eval_dir):
        logging.info(f"[SKIP] {model_dir}: 缺少 demo_images 或 eval_images，跳过")
        return None

    if (not overwrite) and smart_exists(out_json):
        logging.info(f"[SKIP] {model_dir}: {out_json} 已存在（未指定 --overwrite），跳过")
        return None

    demo_imgs = list_images_recursive(demo_dir)
    if not demo_imgs:
        logging.warning(f"[WARN] {model_dir}: demo_images 下没有图片，跳过")
        return None

    # 为了可复现，用 idx+seed 控制本目录的随机选择
    rng = random.Random(base_seed + idx)
    probe_image = rng.choice(demo_imgs)
    logging.info(f"[INFO] {model_dir}: 选中的 probe_image = {probe_image}")

    try:
        result = compute_gallery_probe_with_backends(
            gallery_dir=eval_dir,
            probe_image=probe_image,
            output_json=out_json,
            backends=G_BACKENDS,
            weights=G_WEIGHTS,
            device=G_DEVICE,
        )
        if result is None:
            return None
    except Exception as e:
        logging.error(f"[ERROR] {model_dir}: 计算风格相似度失败（抛异常），跳过该目录 ----> {e}")
        return None

    return out_json


# ==============================================================================
# CLI：遍历 root，多进程处理
# ==============================================================================
def main():
    setup_logging()
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "遍历 root 下子目录：\n"
            "  默认：demo_images 随机一张 vs eval_images gallery mean；\n"
            "  --style-mean：style_100 内每张图 vs 文件夹均值。"
        )
    )

    ap.add_argument(
        "--root",
        required=True,
        help="包含多个 model 子目录的根目录（本地或 s3:// 等桶路径）",
    )
    ap.add_argument(
        "--output-name",
        default="style_similarity.json",
        help="每个子目录下输出 JSON 文件名，默认 style_similarity.json",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="如不指定，则若输出 JSON 已存在则跳过该子目录",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于 demo_images 抽样 probe 或 StyleShot 等）",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="进程数；=1 表示单进程顺序跑；>=2 时使用多进程。",
    )

    # 新增：style_100 均值模式
    ap.add_argument(
        "--style-mean",
        action="store_true",
        help="若指定，则对每个 model_dir 下的 style_100（或 --style-dir-name）目录，"
             "计算『每张图 vs 该文件夹均值』的风格相似度。",
    )
    ap.add_argument(
        "--style-dir-name",
        default="style_100",
        help="style 均值模式下，style 图像子目录名（默认 style_100）",
    )

    # 下面这些参数直接转发给 encoder 构建
    ap.add_argument(
        "--pt_style",
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
        "--encoder_weights",
        default=None,
        help="encoder 权重；两种格式：\n"
             "  1) 与 --pt_style 展开顺序对应的逗号浮点，如 '0.5,0.3,0.2'；\n"
             "  2) 'name:weight,name2:weight2'（name 小写）",
    )

    args = ap.parse_args()
    root = args.root.rstrip("/")
    random.seed(args.seed)

    model_dirs = iter_model_dirs(root)
    if not model_dirs:
        raise SystemExit(f"在 root={root} 下没有找到任何子目录")

    mode_str = "STYLE_MEAN" if args.style_mean else "GALLERY_PROBE"
    logging.info(f"[INFO] 模式 = {mode_str}")
    logging.info(f"[INFO] 在 {root} 下找到 {len(model_dirs)} 个子目录，将逐个计算风格相似度")

    # 展开 pt_style 里的逗号写法
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
        raise SystemExit("请至少指定一个 --pt_style（排除 dino_reg / dinov2_reg_enhanced 后为空）")

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
        "output_name": args.output_name,
        "overwrite": args.overwrite,
        "seed": args.seed,
        "style_mean": args.style_mean,
        "style_dir_name": args.style_dir_name,
    }

    tasks = [(idx, model_dir) for idx, model_dir in enumerate(model_dirs)]

    if args.num_workers <= 1:
        # 单进程：直接在当前进程初始化 backends，然后顺序跑
        logging.info("[INFO] 使用单进程模式")
        init_worker(cfg)
        for t in tasks:
            worker_process(t)
    else:
        # 多进程：每个进程一份 backends
        num_workers = max(1, args.num_workers)
        logging.info(f"[INFO] 使用多进程模式: num_workers={num_workers}")

        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(cfg,),
        ) as pool:
            for _ in pool.imap_unordered(worker_process, tasks):
                pass


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # 可能已经在其他地方设置过
        pass
    main()

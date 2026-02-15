#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gallery mean vs probe similarity tool
- 递归抓取 gallery 目录图片 -> 对每个 encoder 计算 mean 特征向量
- 对 probe 图片提取 encoder 特征 -> 计算与 mean 的相似度
- 支持多个特征后端（和原脚本一致）
- 根据输入的 encoder 权重，对不同模型的相似度做加权求总分

输入：
    --gallery_dir   : 一个包含多张图片的文件夹（递归扫描 png/jpg/jpeg/webp/bmp）
    --probe_image   : 单张要对比的图片路径
    --pt_style      : 使用的特征后端列表（csd,clip,siglip,vgg,sscd,dino,moco,vae,
                      dinov2_adain,styleshot,oneig,dinov2_reg_enhanced 等）
                      本脚本会自动排除 "dino_reg" 和 "dinov2_reg_enhanced"
    --encoder_weights : 各模型权重（可选，见下）
    --output_json   : 输出 JSON 文件路径

encoder_weights 支持两种格式：
    1）与 pt_style 展开后的顺序一一对应的逗号浮点： "0.5,0.3,0.2"
    2）"name:weight,name2:weight2"（name 小写，对应 pt_style 名）：
        例如 "siglip:0.7,oneig:0.3"
    若不指定，则对所有模型均匀分配权重。
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from PIL import Image
import torch
import torch.nn as nn

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
from typing import Dict, Tuple, List, Optional

Image.MAX_IMAGE_PIXELS = 200_000_000
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 需要排除的模型名字（统一小写）
EXCLUDED_MODELS = {"dino_reg", "dinov2_reg_enhanced"}


# ==============================================================================
# 日志 & 设备
# ==============================================================================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# 通用工具：列举文件
# ==============================================================================
def list_images_recursive(root: str) -> List[str]:
    return sorted(str(p) for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


# ==============================================================================
# 一些后端实现（和你原脚本保持一致）
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


def load_and_embed(img_path: str, preprocess, encode,
                   need_fp32: bool, is_dict_input: bool, device: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
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
# Gallery mean & probe 提特征
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
                backends.pop(model_name, None)
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
            backends.pop(model_name, None)
            continue

        feat = feat.detach().cpu().float()
        feat = feat / (feat.norm(p=2) + 1e-8)

        probe_feats[model_name] = feat
        logging.info(f"[{model_name}] probe feat dim={feat.numel()}")

    return probe_feats

def compute_gallery_probe_similarity(
    gallery_dir: str,
    probe_image: str,
    output_json: str = "gallery_probe_similarity.json",
    pt_style: Optional[List[str]] = None,
    arch: str = "vit_base",
    model_path: Optional[str] = None,
    siglip_id: str = None,
    sscd_path: Optional[str] = None,
    moco_ckpt: Optional[str] = None,
    dinov2_id: str = None,
    dinov2_size: int = 518,
    styleshot_clip_path: str = None,
    styleshot_weight_path: str = None,
    oneig_path: str = None,
    encoder_weights: Optional[str] = None,
):
    """
    gallery_mean_vs_probe_with_multi_backends 的函数版本。
    所有原 argparse 参数均通过此函数接收。
    """

    # -------------------------
    #  初始化
    # -------------------------
    setup_logging()

    siglip_id = siglip_id or os.getenv("SIGLIP_PATH", "google/siglip-so400m-patch14-384")
    dinov2_id = dinov2_id or os.getenv("DINOV2_PATH", "facebook/dinov2-base")
    styleshot_clip_path = styleshot_clip_path or os.getenv(
        "STYLESHOT_CLIP", "/mnt/jfs/model_zoo/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    styleshot_weight_path = styleshot_weight_path or os.getenv(
        "STYLESHOT_WEIGHT",
        "/mnt/jfs/model_zoo/StyleShot/StyleShot/pretrained_weight/style_aware_encoder.bin"
    )
    oneig_path = oneig_path or os.getenv("ONEIG_PATH", "xingpng/OneIG-StyleEncoder")

    # -------------------------
    #  整理模型列表
    # -------------------------
    models: List[str] = []
    pt_style = pt_style or []

    for it in pt_style:
        if not it:
            continue
        parts = [p.strip() for p in it.split(",") if p.strip()]
        models.extend(parts)
    models = [m.lower() for m in models]

    if any(m in models for m in EXCLUDED_MODELS):
        logging.info(f"自动移除模型: {EXCLUDED_MODELS & set(models)}")
        models = [m for m in models if m not in EXCLUDED_MODELS]

    if not models:
        raise ValueError("请至少指定一个 pt_style（排除 dino_reg / dinov2_reg_enhanced 后为空）")

    # -------------------------
    #  gallery 图片
    # -------------------------
    gallery_paths = list_images_recursive(gallery_dir)
    if not gallery_paths:
        raise ValueError(f"未在 gallery_dir({gallery_dir}) 中找到图片")

    if not os.path.isfile(probe_image):
        raise ValueError(f"probe_image ({probe_image}) 不存在")

    # -------------------------
    # 构建后端
    # -------------------------
    device = resolve_device()
    logging.info(f"设备: {device}")

    backends: Dict[str, Tuple] = {}
    for pt in models:
        try:
            preprocess, encode, need_fp32, is_dict_input = build_backend(
                pt, 
                type("Args", (), {
                    "arch": arch,
                    "model_path": model_path,
                    "siglip_id": siglip_id,
                    "sscd_path": sscd_path,
                    "moco_ckpt": moco_ckpt,
                    "dinov2_id": dinov2_id,
                    "dinov2_size": dinov2_size,
                    "styleshot_clip_path": styleshot_clip_path,
                    "styleshot_weight_path": styleshot_weight_path,
                    "oneig_path": oneig_path
                })()
            )
            backends[pt] = (preprocess, encode, need_fp32, is_dict_input)
            logging.info(f"[OK] 后端 {pt} 构建成功")
        except Exception as e:
            logging.error(f"[WARN] 后端 {pt} 构建失败，跳过: {e}")

    if not backends:
        raise RuntimeError("没有可用的后端模型，退出")

    # -------------------------
    # 解析 encoder_weights
    # -------------------------
    weights = parse_weights_arg(encoder_weights, list(backends.keys()))
    logging.info(f"encoder weights: {weights}")

    # -------------------------
    #  gallery mean 特征
    # -------------------------
    logging.info("开始对 gallery 计算各 encoder 的 mean 特征...")
    gallery_means, counts = compute_gallery_means(gallery_paths, backends.copy(), device)

    if not gallery_means:
        raise RuntimeError("未能为任何 encoder 计算 gallery mean，退出")

    # -------------------------
    # probe 特征
    # -------------------------
    logging.info("提取 probe 特征...")
    backends_probe = {}
    for pt in gallery_means.keys():
        try:
            preprocess, encode, need_fp32, is_dict_input = build_backend(
                pt,
                type("Args", (), {
                    "arch": arch,
                    "model_path": model_path,
                    "siglip_id": siglip_id,
                    "sscd_path": sscd_path,
                    "moco_ckpt": moco_ckpt,
                    "dinov2_id": dinov2_id,
                    "dinov2_size": dinov2_size,
                    "styleshot_clip_path": styleshot_clip_path,
                    "styleshot_weight_path": styleshot_weight_path,
                    "oneig_path": oneig_path
                })()
            )
            backends_probe[pt] = (preprocess, encode, need_fp32, is_dict_input)
        except Exception as e:
            logging.error(f"[WARN] 构建 probe 后端 {pt} 失败: {e}")

    probe_feats = compute_probe_feats(probe_image, backends_probe.copy(), device)

    # -------------------------
    # per-model 计算相似度
    # -------------------------
    per_model: Dict[str, Dict[str, float]] = {}
    for model_name, mean_feat in gallery_means.items():
        if model_name not in probe_feats:
            logging.warning(f"[{model_name}] probe 未提取到特征，跳过")
            continue

        sim = cosine_sim(mean_feat, probe_feats[model_name])
        per_model[model_name] = {
            "similarity": float(sim),
            "gallery_count": int(counts.get(model_name, 0)),
        }

    # -------------------------
    # 加权平均
    # -------------------------
    total = 0.0
    weight_sum = 0.0
    for model_name, info in per_model.items():
        w = weights.get(model_name, 0.0)
        total += w * info["similarity"]
        weight_sum += w

    weighted_score = total / weight_sum if weight_sum > 0 else None

    # -------------------------
    # 输出结构
    # -------------------------
    output = {
        "probe_image": os.path.abspath(probe_image),
        "gallery_dir": os.path.abspath(gallery_dir),
        "per_model": per_model,
        "encoder_weights_used": weights,
        "weighted_score": None if weighted_score is None else float(weighted_score),
    }

    outp = Path(output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logging.info(f"[DONE] 输出 JSON: {outp}")

    return output

# ==============================================================================
# CLI
# ==============================================================================
def main():
    setup_logging()
    import argparse

    ap = argparse.ArgumentParser("gallery_mean_vs_probe_with_multi_backends")

    # I/O
    ap.add_argument("--gallery_dir", required=True, help="包含多张 gallery 图片的根目录（递归）")
    ap.add_argument("--probe_image", required=True, help="单张 probe 图片路径")
    ap.add_argument("--output_json", default="gallery_probe_similarity.json",
                    help="输出 JSON 路径")

    # 后端参数（和你原脚本一致）
    ap.add_argument("--pt_style", action="append", default=[],
                    help="可多次传入或用逗号分隔：csd,clip,siglip,vgg,sscd,dino,moco,vae,"
                         "dinov2_adain,styleshot,oneig,dinov2_reg_enhanced")
    ap.add_argument("--arch", default="vit_base")
    ap.add_argument("--model_path", default=None, help="csd/vae 等需要的权重路径")
    ap.add_argument("--siglip_id", default=os.getenv("SIGLIP_PATH", "google/siglip-so400m-patch14-384"))
    ap.add_argument("--sscd_path", default=None)
    ap.add_argument("--moco_ckpt", default=None)
    ap.add_argument("--dinov2_id", default=os.getenv("DINOV2_PATH", "facebook/dinov2-base"))
    ap.add_argument("--dinov2_size", type=int, default=518)
    ap.add_argument("--styleshot_clip_path",
                    default=os.getenv("STYLESHOT_CLIP", "/mnt/jfs/model_zoo/CLIP-ViT-H-14-laion2B-s32B-b79K"))
    ap.add_argument("--styleshot_weight_path",
                    default=os.getenv("STYLESHOT_WEIGHT",
                                      "/mnt/jfs/model_zoo/StyleShot/StyleShot/pretrained_weight/style_aware_encoder.bin"))
    ap.add_argument("--oneig_path", default=os.getenv("ONEIG_PATH", "xingpng/OneIG-StyleEncoder"))

    # encoder 权重
    ap.add_argument(
        "--encoder_weights",
        default=None,
        help="encoder 权重；两种格式："
             "1) 与 --pt_style 展开顺序对应的逗号浮点，如 '0.5,0.3,0.2'；"
             "2) 'name:weight,name2:weight2'（name 小写）",
    )

    args = ap.parse_args()

    # 展开 pt_style 中的逗号形式
    models: List[str] = []
    for it in args.pt_style:
        if not it:
            continue
        parts = [p.strip() for p in it.split(",") if p.strip()]
        models.extend(parts)
    models = [m.lower() for m in models]

    # 自动排除 dino_reg / dinov2_reg_enhanced
    if any(m in models for m in EXCLUDED_MODELS):
        logging.info(f"自动移除模型: {EXCLUDED_MODELS & set(models)}")
        models = [m for m in models if m not in EXCLUDED_MODELS]

    if not models:
        raise SystemExit("请至少指定一个 --pt_style（排除 dino_reg / dinov2_reg_enhanced 后为空）")

    # gallery 图片列表
    gallery_paths = list_images_recursive(args.gallery_dir)
    if not gallery_paths:
        raise SystemExit(f"未在 gallery_dir({args.gallery_dir}) 中找到图片")

    probe_path = args.probe_image
    if not os.path.isfile(probe_path):
        raise SystemExit(f"probe_image ({probe_path}) 不存在")

    device = resolve_device()
    logging.info(f"设备: {device}")

    # 构建后端
    backends: Dict[str, Tuple] = {}
    for pt in models:
        try:
            preprocess, encode, need_fp32, is_dict_input = build_backend(pt, args)
            backends[pt] = (preprocess, encode, need_fp32, is_dict_input)
            logging.info(f"[OK] 后端 {pt} 构建成功")
        except Exception as e:
            logging.error(f"[WARN] 后端 {pt} 构建失败，跳过: {e}")

    if not backends:
        raise SystemExit("没有可用的后端模型，退出")

    # 解析 encoder 权重
    weights = parse_weights_arg(args.encoder_weights, list(backends.keys()))
    logging.info(f"encoder weights: {weights}")

    # 计算 gallery mean 特征
    logging.info("开始对 gallery 计算各 encoder 的 mean 特征（可能较慢）...")
    gallery_means, counts = compute_gallery_means(gallery_paths, backends.copy(), device)

    if not gallery_means:
        raise SystemExit("未能为任何 encoder 计算 gallery mean，退出")

    # 为 probe 构建与 gallery_means 同集合的 backends
    active_models = list(gallery_means.keys())
    backends_probe: Dict[str, Tuple] = {}
    for pt in active_models:
        try:
            preprocess, encode, need_fp32, is_dict_input = build_backend(pt, args)
            backends_probe[pt] = (preprocess, encode, need_fp32, is_dict_input)
        except Exception as e:
            logging.error(f"[WARN] 构建 probe 后端 {pt} 失败: {e}")

    # 提取 probe 特征
    logging.info("提取 probe 特征...")
    probe_feats = compute_probe_feats(probe_path, backends_probe.copy(), device)

    # per-model 相似度
    per_model: Dict[str, Dict[str, float]] = {}
    for model_name, mean_feat in gallery_means.items():
        if model_name not in probe_feats:
            logging.warning(f"[{model_name}] probe 未提取到特征，跳过相似度计算")
            continue
        pfeat = probe_feats[model_name]
        sim = cosine_sim(mean_feat, pfeat)
        per_model[model_name] = {
            "similarity": float(sim),
            "gallery_count": int(counts.get(model_name, 0)),
        }

    # 加权总分
    total = 0.0
    weight_sum = 0.0
    for model_name, info in per_model.items():
        w = weights.get(model_name, 0.0)
        total += w * info["similarity"]
        weight_sum += w
    weighted_score = total / weight_sum if weight_sum > 0 else None

    output = {
        "probe_image": os.path.abspath(probe_path),
        "gallery_dir": os.path.abspath(args.gallery_dir),
        "per_model": per_model,
        "encoder_weights_used": weights,
        "weighted_score": None if weighted_score is None else float(weighted_score),
    }

    outp = Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logging.info(f"[DONE] 输出 JSON: {outp}")


if __name__ == "__main__":
    main()

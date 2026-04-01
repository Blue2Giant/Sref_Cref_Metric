#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双流的attention中：q是[text | latent | cref | sref]，k是[text | latent | cref | sref]
单流的attention中：q是[text | latent | cref | sref]，k是[text | latent | cref | sref]
0 到 7 是双流块 Flux2Attention
8 到 55 是单流块 Flux2ParallelSelfAttention
"""

import argparse
import json
import math
import multiprocessing as mp
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Flux2-Klein-9B Attention FullMap 可视化")
    parser.add_argument("--prompts_json", required=True, help="id->prompt 的json文件")
    parser.add_argument("--cref_dir", required=True, help="内容参考图目录，文件名应为 {id}.png")
    parser.add_argument("--sref_dir", required=True, help="风格参考图目录，文件名应为 {id}.png")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--gpus", default="0", help='如 "0" 或 "0,1,2"，会在这些GPU上并行推理')
    parser.add_argument("--key_txt", required=True, help="txt文件，可包含多行key")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128, help="attention矩阵query最大采样token数")
    parser.add_argument("--max-q-tokens", type=int, default=0, help="query采样token上限，<=0时使用max-tokens")
    parser.add_argument("--max-k-tokens", type=int, default=0, help="兼容保留参数，当前脚本会保留全部key token")
    parser.add_argument("--aggregate-head", choices=["mean", "max"], default="mean")
    parser.add_argument("--aggregate-block", choices=["mean", "max"], default="mean")
    parser.add_argument("--step-stride", type=int, default=4, help="每隔多少步采样一个step")
    parser.add_argument("--block-stride", type=int, default=4, help="每隔多少个block采样一个")
    parser.add_argument("--panel-size", type=float, default=1.4, help="step-block拼图每个子图尺寸")
    parser.add_argument("--image-dpi", type=int, default=120)
    parser.add_argument("--save-format", choices=["png", "pdf"], default="png")
    parser.add_argument("--ref-labels", default="cref,sref", help="参考图名称，逗号分隔，按输入图顺序")
    parser.add_argument("--attn-gamma", type=float, default=0.6, help="attention强度gamma增强，越小越强调高值")
    parser.add_argument("--boundary-linewidth", type=float, default=1.0, help="参考图分界线宽度")
    parser.add_argument("--save-attn-tensor", action="store_true", help="额外保存attention张量为pt文件")
    parser.add_argument("--cache-step-block", action="store_true", help="按step/block缓存attention到磁盘，降低内存占用")
    parser.add_argument(
        "--cache-full-attn",
        action="store_true",
        help="按step/block缓存完整attention张量用于分析；Q维不做采样，save-attn-tensor将保存manifest",
    )
    parser.add_argument("--show-inner-progress", action="store_true", help="显示diffusers内部step进度条")
    parser.add_argument("--input_resolution", type=str, default="", help='统一覆盖cref分辨率，如 "1024x1024"')
    parser.add_argument(
        "--max-input-long-side",
        type=int,
        default=0,
        help="如果>0，先将每张输入图按长边等比缩小到不超过该值，避免OOM；不会放大较小图片",
    )
    parser.add_argument("--no_resize_cref", action="store_true", help="关闭cref按预设宽高比resize")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument(
        "--text_encoder_out_layers",
        type=str,
        default="9,18,27",
        help='Comma-separated Qwen layer indices, e.g. "9,18,27".',
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


PREFERRED_KONTEXT_RESOLUTIONS: List[Tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


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


def read_keys(key_txt: str) -> List[str]:
    keys = []
    seen = set()
    with open(key_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if s and s not in seen:
                keys.append(s)
                seen.add(s)
    if not keys:
        raise RuntimeError(f"key_txt中没有可用key: {key_txt}")
    return keys


def _lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def resize_cref(cref: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
    cref_w, cref_h = cref.size
    aspect_ratio = cref_w / float(cref_h)

    _, target_w, target_h = min(
        (abs(aspect_ratio - (w / float(h))), w, h) for (w, h) in PREFERRED_KONTEXT_RESOLUTIONS
    )

    if (cref_w, cref_h) == (target_w, target_h):
        return cref, (target_w, target_h)

    resized = cref.resize((target_w, target_h), resample=_lanczos())
    return resized, (target_w, target_h)


def resize_long_side_limit(img: Image.Image, max_long_side: int) -> Image.Image:
    limit = int(max_long_side)
    if limit <= 0:
        return img
    width, height = img.size
    long_side = max(int(width), int(height))
    if long_side <= limit:
        return img
    scale = float(limit) / float(long_side)
    new_width = max(1, int(round(float(width) * scale)))
    new_height = max(1, int(round(float(height) * scale)))
    if (new_width, new_height) == (width, height):
        return img
    return img.resize((new_width, new_height), resample=_lanczos())


def choose_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def parse_text_encoder_out_layers(arg: str) -> Tuple[int, ...]:
    values = [x.strip() for x in str(arg).split(",") if x.strip()]
    if not values:
        return (9, 18, 27)
    return tuple(int(x) for x in values)


def parse_gpu_list(gpus_arg: str) -> List[int]:
    raw = [x.strip() for x in str(gpus_arg).split(",") if x.strip()]
    out: List[int] = []
    seen = set()
    for x in raw:
        gpu = int(x)
        if gpu < 0:
            raise RuntimeError(f"GPU编号必须>=0，收到: {gpu}")
        if gpu not in seen:
            out.append(gpu)
            seen.add(gpu)
    if not out:
        out = [0]
    return out


def split_keys_round_robin(keys: List[str], n_parts: int) -> List[List[str]]:
    n = max(1, int(n_parts))
    out: List[List[str]] = [[] for _ in range(n)]
    for idx, key in enumerate(keys):
        out[idx % n].append(key)
    return out


def normalize_size_wh(width: int, height: int, multiple: int) -> Tuple[int, int]:
    width = max(multiple, (int(width) // multiple) * multiple)
    height = max(multiple, (int(height) // multiple) * multiple)
    return width, height


def compute_flux_token_count(width: int, height: int, vae_scale_factor: int) -> int:
    multiple = int(vae_scale_factor) * 2
    width, height = normalize_size_wh(width, height, multiple)
    return int((width // multiple) * (height // multiple))


def build_ref_labels(raw_labels: str, num_refs: int) -> List[str]:
    labels = [x.strip() for x in str(raw_labels).split(",") if x.strip()]
    if len(labels) < num_refs:
        labels += [f"ref{i}" for i in range(len(labels), num_refs)]
    return labels[:num_refs]


def build_joint_token_ranges(
    text_tokens_full: int,
    latent_tokens_full: int,
    ref_infos: List[Dict[str, Any]],
) -> List[Tuple[str, int, int]]:
    cursor = 0
    ranges: List[Tuple[str, int, int]] = []
    if int(text_tokens_full) > 0:
        ranges.append(("text", cursor, cursor + int(text_tokens_full)))
        cursor += int(text_tokens_full)
    if int(latent_tokens_full) > 0:
        ranges.append(("latent", cursor, cursor + int(latent_tokens_full)))
        cursor += int(latent_tokens_full)
    for info in ref_infos:
        label = str(info.get("label", "ref")).strip() or "ref"
        token_count = max(0, int(info.get("token_count", 0)))
        if token_count > 0:
            ranges.append((label, cursor, cursor + token_count))
            cursor += token_count
    return ranges


def build_range_metadata(
    ranges: List[Tuple[str, int, int]],
    sample_indices: List[int],
) -> List[Dict[str, Any]]:
    arr = np.asarray(sample_indices or [], dtype=np.int64)
    out: List[Dict[str, Any]] = []
    for name, fs, fe in ranges:
        item: Dict[str, Any] = {
            "name": str(name),
            "full_start": int(fs),
            "full_end_exclusive": int(fe),
            "full_end_inclusive": int(fe - 1),
            "full_length": max(int(fe) - int(fs), 0),
            "sample_start": None,
            "sample_end_exclusive": None,
            "sample_end_inclusive": None,
            "sample_length": 0,
        }
        if arr.size > 0:
            loc = np.where((arr >= int(fs)) & (arr < int(fe)))[0]
            if loc.size > 0:
                ss = int(loc.min())
                se = int(loc.max()) + 1
                item["sample_start"] = ss
                item["sample_end_exclusive"] = se
                item["sample_end_inclusive"] = se - 1
                item["sample_length"] = se - ss
        out.append(item)
    return out


def build_range_metadata_by_name(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        name = str(item.get("name", "")).strip()
        if name:
            out[name] = dict(item)
    return out


def map_ranges_to_sample_spans(
    ranges: List[Tuple[str, int, int]],
    sample_indices: List[int],
) -> List[Tuple[str, int, int]]:
    if not ranges or not sample_indices:
        return []
    arr = np.asarray(sample_indices, dtype=np.int64)
    mapped: List[Tuple[str, int, int]] = []
    for name, fs, fe in ranges:
        loc = np.where((arr >= int(fs)) & (arr < int(fe)))[0]
        if loc.size <= 0:
            continue
        ss = int(loc.min())
        se = int(loc.max()) + 1
        if se > ss:
            mapped.append((str(name), ss, se))
    return mapped


def _segment_base_color(name: str) -> np.ndarray:
    s = str(name or "").lower()
    if "cref" in s:
        return np.array([0.35, 0.06, 0.06], dtype=np.float32)
    if "sref" in s:
        return np.array([0.07, 0.12, 0.35], dtype=np.float32)
    if "text" in s:
        return np.array([0.05, 0.24, 0.10], dtype=np.float32)
    if "latent" in s or "noise" in s:
        return np.array([0.18, 0.18, 0.18], dtype=np.float32)
    return np.array([0.14, 0.14, 0.20], dtype=np.float32)


def power_normalize(arr: np.ndarray, vmin: float, vmax: float, gamma: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    denom = max(float(vmax) - float(vmin), 1e-6)
    x = np.clip((arr - float(vmin)) / denom, 0.0, 1.0)
    return np.power(x, max(float(gamma), 1e-3), dtype=np.float32)


def render_segmented_attn_rgb(
    mat_np: np.ndarray,
    norm_fn,
    k_spans: List[Tuple[str, int, int]],
) -> np.ndarray:
    h, w = mat_np.shape
    base_cols = np.tile(np.array([0.14, 0.14, 0.20], dtype=np.float32), (w, 1))
    for name, s, e in k_spans:
        ss = max(0, int(s))
        ee = min(w, int(e))
        if ee > ss:
            base_cols[ss:ee, :] = _segment_base_color(name)
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    v = np.asarray(norm_fn(mat_np), dtype=np.float32)
    v = np.clip(v, 0.0, 1.0)[..., None]
    rgb = base_cols[None, :, :] * (1.0 - v) + yellow[None, None, :] * v
    return np.clip(rgb, 0.0, 1.0)


def draw_span_boundaries_on_rgb(
    rgb: np.ndarray,
    q_spans: List[Tuple[str, int, int]],
    k_spans: List[Tuple[str, int, int]],
    boundary_linewidth: float,
) -> np.ndarray:
    out = np.array(rgb, copy=True)
    h, w, _ = out.shape
    lw = max(1, int(round(float(boundary_linewidth))))
    for _name, _s, e in k_spans[:-1]:
        col = min(max(int(e) - 1, 0), w - 1)
        left = max(0, col - lw // 2)
        right = min(w, left + lw)
        out[:, left:right, :] = 0.0
    for _name, _s, e in q_spans[:-1]:
        row = min(max(int(e) - 1, 0), h - 1)
        top = max(0, row - lw // 2)
        bottom = min(h, top + lw)
        out[top:bottom, :, :] = 0.0
    return out


def _summarize_indices(indices: List[int], limit: int = 16) -> str:
    if not indices:
        return "[]"
    if len(indices) <= limit:
        return "[" + ", ".join(str(int(x)) for x in indices) + "]"
    head = ", ".join(str(int(x)) for x in indices[: limit // 2])
    tail = ", ".join(str(int(x)) for x in indices[-(limit // 2) :])
    return f"[{head}, ..., {tail}] (n={len(indices)})"


def _is_nonempty_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return int(path.stat().st_size) > 0
    except Exception:
        return False


def _is_valid_output_file(path: Path, save_format: str) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        size = int(path.stat().st_size)
    except Exception:
        return False
    if size <= 1024:
        return False
    sf = str(save_format).lower().strip()
    try:
        with path.open("rb") as f:
            head = f.read(16)
        if sf == "png":
            return head.startswith(b"\x89PNG\r\n\x1a\n")
        if sf == "pdf":
            return head.startswith(b"%PDF")
    except Exception:
        return False
    return True


def _has_complete_attention_outputs(
    key_attn_dir: Path,
    out_img: Path,
    save_format: str,
    save_attn_tensor: bool = False,
) -> bool:
    out_map = key_attn_dir / f"attention_step_block_grid.{save_format}"
    out_summary = key_attn_dir / "attention_region_summary.json"
    out_layout = key_attn_dir / "token_layout_summary.txt"
    if not _is_valid_output_file(out_map, save_format):
        return False
    if not _is_nonempty_file(out_img):
        return False
    if not _is_nonempty_file(out_summary) or not _is_nonempty_file(out_layout):
        return False
    if bool(save_attn_tensor):
        out_tensor = key_attn_dir / "attention_step_block_grid.pt"
        if not _is_nonempty_file(out_tensor):
            return False
    return True


def _get_apply_rotary_emb():
    from diffusers.models.embeddings import apply_rotary_emb

    return apply_rotary_emb


class FluxAttentionCollector:
    def __init__(
        self,
        max_q_tokens: int,
        aggregate_head: str = "mean",
        block_stride: int = 1,
        keep_full_query: bool = False,
        owner_label: str = "",
    ):
        self.max_q_tokens = int(max_q_tokens)
        self.aggregate_head = str(aggregate_head)
        self.block_stride = max(1, int(block_stride))
        self.keep_full_query = bool(keep_full_query)
        self.owner_label = str(owner_label)
        self.handles = []
        self.maps: Dict[int, torch.Tensor] = {}
        self.shared_meta: Dict[str, Any] = {}
        self.block_info: Dict[int, Dict[str, Any]] = {}
        self.hook_error_count = 0
        self.first_hook_error: Optional[Dict[str, Any]] = None

    def _record_hook_error(self, block_idx: int, module: torch.nn.Module, exc: Exception):
        self.hook_error_count += 1
        if self.first_hook_error is not None:
            return
        info = {
            "block_idx": int(block_idx),
            "module_name": module.__class__.__name__,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.first_hook_error = info
        prefix = self.owner_label or "collector"
        print(
            f"[ATTN-HOOK-ERROR]{prefix} block={info['block_idx']} module={info['module_name']} "
            f"{info['error_type']}: {info['error_message']}"
            ,
            flush=True,
        )
        tb = str(info["traceback"]).rstrip()
        if tb:
            print(tb, flush=True)

    def describe_first_hook_error(self) -> str:
        if self.first_hook_error is None:
            return ""
        info = self.first_hook_error
        return (
            f"block={info['block_idx']} module={info['module_name']} "
            f"{info['error_type']}: {info['error_message']}"
        )

    def _sample_tokens(self, n: int, device: torch.device) -> torch.Tensor:
        if self.keep_full_query or int(self.max_q_tokens) <= 0:
            return torch.arange(n, device=device, dtype=torch.long)
        if n <= self.max_q_tokens:
            return torch.arange(n, device=device, dtype=torch.long)
        idx = torch.linspace(0, n - 1, steps=self.max_q_tokens, device=device)
        idx = torch.round(idx).long()
        idx = torch.unique_consecutive(idx)
        if idx.numel() < self.max_q_tokens:
            padding = torch.arange(n, device=device, dtype=torch.long)
            idx = torch.unique_consecutive(torch.cat([idx, padding], dim=0))[: self.max_q_tokens]
        return idx

    def _compute_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        q_idx: torch.Tensor,
    ) -> torch.Tensor:
        query = query[:, q_idx]
        query = query.permute(0, 2, 1, 3).float()
        key = key.permute(0, 2, 1, 3).float()
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(query.shape[-1]))
        attn = torch.softmax(scores, dim=-1)
        if self.aggregate_head == "max":
            attn = attn.max(dim=1).values
        else:
            attn = attn.mean(dim=1)
        return attn[0].detach().cpu().to(torch.float16)

    def _record_shared_meta(self, q_full: int, k_full: int, q_idx: torch.Tensor):
        if self.shared_meta:
            return
        self.shared_meta = {
            "q_tokens_full": int(q_full),
            "k_tokens_full": int(k_full),
            "q_sample_indices": [int(x) for x in q_idx.detach().cpu().tolist()],
            "k_sample_indices": list(range(int(k_full))),
        }

    def _hook_flux2_attention(self, block_idx: int):
        def _fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if encoder_hidden_states is None and len(args) > 1:
                encoder_hidden_states = args[1]
            image_rotary_emb = kwargs.get("image_rotary_emb")
            if hidden_states is None or not torch.is_tensor(hidden_states):
                return
            try:
                if getattr(module, "fused_projections", False) and hasattr(module, "to_qkv"):
                    query, key, _value = module.to_qkv(hidden_states).chunk(3, dim=-1)
                    encoder_query = encoder_key = None
                    if encoder_hidden_states is not None and hasattr(module, "to_added_qkv"):
                        encoder_query, encoder_key, _encoder_value = module.to_added_qkv(encoder_hidden_states).chunk(
                            3, dim=-1
                        )
                else:
                    query = module.to_q(hidden_states)
                    key = module.to_k(hidden_states)
                    encoder_query = encoder_key = None
                    if encoder_hidden_states is not None and getattr(module, "added_kv_proj_dim", None) is not None:
                        encoder_query = module.add_q_proj(encoder_hidden_states)
                        encoder_key = module.add_k_proj(encoder_hidden_states)

                query = query.unflatten(-1, (module.heads, -1))
                key = key.unflatten(-1, (module.heads, -1))
                query = module.norm_q(query)
                key = module.norm_k(key)

                if encoder_query is not None and encoder_key is not None:
                    encoder_query = encoder_query.unflatten(-1, (module.heads, -1))
                    encoder_key = encoder_key.unflatten(-1, (module.heads, -1))
                    encoder_query = module.norm_added_q(encoder_query)
                    encoder_key = module.norm_added_k(encoder_key)
                    query = torch.cat([encoder_query, query], dim=1)
                    key = torch.cat([encoder_key, key], dim=1)

                if image_rotary_emb is not None:
                    apply_rotary_emb = _get_apply_rotary_emb()
                    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

                q_idx = self._sample_tokens(int(query.shape[1]), query.device)
                self._record_shared_meta(int(query.shape[1]), int(key.shape[1]), q_idx)
                self.maps[block_idx] = self._compute_scores(query, key, q_idx)
            except Exception as exc:
                self._record_hook_error(block_idx, module, exc)
                return

        return _fn

    def _hook_flux2_parallel_self_attention(self, block_idx: int):
        def _fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            image_rotary_emb = kwargs.get("image_rotary_emb")
            if hidden_states is None or not torch.is_tensor(hidden_states):
                return
            try:
                qkv_mlp = module.to_qkv_mlp_proj(hidden_states)
                qkv, _mlp_hidden_states = torch.split(
                    qkv_mlp,
                    [3 * module.inner_dim, module.mlp_hidden_dim * module.mlp_mult_factor],
                    dim=-1,
                )
                query, key, _value = qkv.chunk(3, dim=-1)
                query = query.unflatten(-1, (module.heads, -1))
                key = key.unflatten(-1, (module.heads, -1))
                query = module.norm_q(query)
                key = module.norm_k(key)

                if image_rotary_emb is not None:
                    apply_rotary_emb = _get_apply_rotary_emb()
                    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

                q_idx = self._sample_tokens(int(query.shape[1]), query.device)
                self._record_shared_meta(int(query.shape[1]), int(key.shape[1]), q_idx)
                self.maps[block_idx] = self._compute_scores(query, key, q_idx)
            except Exception as exc:
                self._record_hook_error(block_idx, module, exc)
                return

        return _fn

    def register(self, model: torch.nn.Module):
        block_idx = 0
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            if cls_name not in {"Flux2Attention", "Flux2ParallelSelfAttention"}:
                continue

            kind = "double_stream" if cls_name == "Flux2Attention" else "single_stream"
            self.block_info[block_idx] = {"name": str(name), "kind": kind}
            if block_idx % self.block_stride == 0:
                hook_fn = (
                    self._hook_flux2_attention(block_idx)
                    if cls_name == "Flux2Attention"
                    else self._hook_flux2_parallel_self_attention(block_idx)
                )
                self.handles.append(module.register_forward_hook(hook_fn, with_kwargs=True))
            block_idx += 1

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def filter_block_info(
    block_info: Dict[int, Dict[str, Any]],
    block_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    keep = {int(x) for x in block_ids}
    return {int(k): dict(v) for k, v in block_info.items() if int(k) in keep}


def _load_step_block_entry(entry: Any) -> Dict[int, torch.Tensor]:
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, (str, Path)):
        data = torch.load(entry, map_location="cpu")
        if isinstance(data, dict):
            return data
    raise TypeError(f"Unsupported step-block entry type: {type(entry)}")


def _collect_step_and_block_ids(step_block_maps: Dict[int, Any]) -> Tuple[List[int], List[int]]:
    if not step_block_maps:
        raise RuntimeError("没有step-block attention")
    steps = sorted(int(x) for x in step_block_maps.keys())
    block_ids = set()
    for step in steps:
        cur = _load_step_block_entry(step_block_maps[step])
        for block_id in cur.keys():
            block_ids.add(int(block_id))
    block_ids = sorted(block_ids)
    if not block_ids:
        raise RuntimeError("没有attention block")
    return steps, block_ids


def pack_step_block_maps(
    step_block_maps: Dict[int, Any]
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    steps, block_ids = _collect_step_and_block_ids(step_block_maps)

    q_target = 0
    k_target = 0
    for step in steps:
        cur_map = _load_step_block_entry(step_block_maps[step])
        for block_id in block_ids:
            mat = cur_map.get(block_id)
            if mat is None:
                continue
            q_target = max(q_target, int(mat.shape[0]))
            k_target = max(k_target, int(mat.shape[1]))
    if q_target <= 0 or k_target <= 0:
        raise RuntimeError("attention为空")

    tensor = torch.full((len(steps), len(block_ids), q_target, k_target), float("nan"), dtype=torch.float32)
    mask = torch.zeros((len(steps), len(block_ids)), dtype=torch.bool)
    for sidx, step in enumerate(steps):
        cur_map = _load_step_block_entry(step_block_maps[step])
        for bidx, block_id in enumerate(block_ids):
            mat = cur_map.get(block_id)
            if mat is None:
                continue
            cur = mat.detach().float().cpu()
            if int(cur.shape[0]) != q_target or int(cur.shape[1]) != k_target:
                cur = torch.nn.functional.interpolate(
                    cur.unsqueeze(0).unsqueeze(0),
                    size=(q_target, k_target),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            tensor[sidx, bidx] = cur
            mask[sidx, bidx] = True
    return tensor, mask, steps, block_ids


def reduce_mats(mats: List[torch.Tensor], mode: str) -> torch.Tensor:
    if not mats:
        raise RuntimeError("没有attention用于聚合")
    reduced = mats[0].detach().float().cpu().clone()
    if str(mode) == "max":
        for x in mats[1:]:
            reduced = torch.maximum(reduced, x.detach().float().cpu())
        return reduced
    count = 1
    for x in mats[1:]:
        reduced = reduced + x.detach().float().cpu()
        count += 1
    return reduced / float(max(count, 1))


def compute_region_mass_matrix(
    mat: torch.Tensor,
    q_spans: List[Tuple[str, int, int]],
    k_spans: List[Tuple[str, int, int]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    mat = mat.detach().float().cpu()
    for q_name, qs, qe in q_spans:
        if qe <= qs:
            continue
        q_chunk = mat[qs:qe]
        if q_chunk.numel() == 0:
            continue
        row: Dict[str, float] = {}
        for k_name, ks, ke in k_spans:
            if ke <= ks:
                continue
            score = q_chunk[:, ks:ke].sum(dim=1).mean().item()
            row[str(k_name)] = float(score)
        out[str(q_name)] = row
    return out


def save_region_summary(
    step_block_maps: Dict[int, Any],
    out_path: Path,
    q_ranges: List[Tuple[str, int, int]],
    k_ranges: List[Tuple[str, int, int]],
    q_sample_indices: List[int],
    k_sample_indices: List[int],
    block_info: Dict[int, Dict[str, Any]],
    aggregate_block: str,
):
    q_spans = map_ranges_to_sample_spans(q_ranges, q_sample_indices)
    k_spans = map_ranges_to_sample_spans(k_ranges, k_sample_indices)
    steps, block_ids = _collect_step_and_block_ids(step_block_maps)
    sampled_block_info = filter_block_info(block_info, block_ids)

    per_step: Dict[str, Dict[str, Dict[str, float]]] = {}
    overall = None
    overall_count = 0
    for step in steps:
        cur_map = _load_step_block_entry(step_block_maps[step])
        step_mats = [cur_map[b] for b in block_ids if b in cur_map]
        if not step_mats:
            continue
        reduced = reduce_mats(step_mats, mode=aggregate_block)
        per_step[str(step)] = compute_region_mass_matrix(reduced, q_spans, k_spans)
        for mat in step_mats:
            cur = mat.detach().float().cpu()
            if overall is None:
                overall = cur.clone()
            elif str(aggregate_block) == "max":
                overall = torch.maximum(overall, cur)
            else:
                overall = overall + cur
            overall_count += 1

    if overall is None or overall_count <= 0:
        raise RuntimeError("没有attention用于region summary")
    if str(aggregate_block) != "max":
        overall = overall / float(overall_count)
    payload = {
        "aggregate_block": str(aggregate_block),
        "q_sample_spans": [{"name": name, "sample_start": s, "sample_end_exclusive": e} for name, s, e in q_spans],
        "k_sample_spans": [{"name": name, "sample_start": s, "sample_end_exclusive": e} for name, s, e in k_spans],
        "overall_query_to_key_mass": compute_region_mass_matrix(overall, q_spans, k_spans),
        "per_step_query_to_key_mass": per_step,
        "block_info": {str(k): dict(v) for k, v in sampled_block_info.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sample_flat_tensor(flat: torch.Tensor, max_points: int) -> torch.Tensor:
    flat = flat.reshape(-1).detach().float().cpu()
    if int(flat.numel()) <= int(max_points):
        return flat
    idx = torch.linspace(0, int(flat.numel()) - 1, steps=int(max_points), dtype=torch.float64)
    idx = torch.round(idx).long()
    idx = torch.unique_consecutive(idx)
    if int(idx.numel()) < int(max_points):
        pad = torch.arange(int(flat.numel()), dtype=torch.long)
        idx = torch.unique_consecutive(torch.cat([idx, pad], dim=0))[: int(max_points)]
    return flat[idx]


def _estimate_quantiles_from_flat_sample(
    flat: torch.Tensor,
    q_low: float,
    q_high: float,
    fallback_min: Optional[float] = None,
    fallback_max: Optional[float] = None,
) -> Tuple[float, float]:
    arr = np.asarray(flat.reshape(-1).detach().cpu().numpy(), dtype=np.float32)
    if arr.size <= 0:
        raise RuntimeError("attention sample 为空")
    vmin = float(np.quantile(arr, float(q_low)))
    vmax = float(np.quantile(arr, float(q_high)))
    if vmax <= vmin:
        low = float(fallback_min if fallback_min is not None else float(arr.min()))
        high = float(fallback_max if fallback_max is not None else float(arr.max()))
        if high <= low:
            high = low + 1e-6
        return low, high
    return vmin, vmax


def _estimate_value_range_from_step_block_maps(
    step_block_maps: Dict[int, Any],
    q_low: float = 0.02,
    q_high: float = 0.995,
    max_points: int = 1_000_000,
) -> Tuple[float, float]:
    steps, block_ids = _collect_step_and_block_ids(step_block_maps)
    total_entries = 0
    for step in steps:
        cur_map = _load_step_block_entry(step_block_maps[step])
        total_entries += max(1, len(cur_map))

    samples: List[torch.Tensor] = []
    global_min = None
    global_max = None
    remaining_budget = max(1, int(max_points))
    remaining_entries = max(1, int(total_entries))
    for step in steps:
        cur_map = _load_step_block_entry(step_block_maps[step])
        for block_id in block_ids:
            mat = cur_map.get(block_id)
            if mat is None:
                continue
            cur_min = float(mat.min().item())
            cur_max = float(mat.max().item())
            global_min = cur_min if global_min is None else min(global_min, cur_min)
            global_max = cur_max if global_max is None else max(global_max, cur_max)
            per_entry_limit = int(math.ceil(float(remaining_budget) / float(max(remaining_entries, 1))))
            if per_entry_limit > 0:
                sampled = _sample_flat_tensor(mat, per_entry_limit)
                if int(sampled.numel()) > 0:
                    samples.append(sampled)
                    remaining_budget = max(0, int(remaining_budget) - int(sampled.numel()))
            remaining_entries = max(0, int(remaining_entries) - 1)

    if not samples:
        raise RuntimeError("没有attention可视化")

    flat = torch.cat(samples, dim=0)
    return _estimate_quantiles_from_flat_sample(
        flat,
        q_low=q_low,
        q_high=q_high,
        fallback_min=global_min,
        fallback_max=global_max,
    )


def save_step_block_grid(
    step_block_maps: Dict[int, Any],
    out_path: Path,
    q_ranges: List[Tuple[str, int, int]],
    k_ranges: List[Tuple[str, int, int]],
    q_sample_indices: List[int],
    k_sample_indices: List[int],
    image_dpi: int,
    save_format: str,
    step_stride: int,
    block_stride: int,
    panel_size: float,
    attn_gamma: float,
    boundary_linewidth: float,
):
    steps, block_ids = _collect_step_and_block_ids(step_block_maps)
    vmin, vmax = _estimate_value_range_from_step_block_maps(step_block_maps)
    norm_fn = lambda x: power_normalize(x, vmin=vmin, vmax=vmax, gamma=float(attn_gamma))

    q_spans = map_ranges_to_sample_spans(q_ranges, q_sample_indices)
    k_spans = map_ranges_to_sample_spans(k_ranges, k_sample_indices)

    sample_map = _load_step_block_entry(step_block_maps[steps[0]])
    sample_mat = sample_map[block_ids[0]].detach().float().cpu().numpy()
    q_len, k_len = sample_mat.shape
    base_panel_h = max(96, int(float(panel_size) * 110.0))
    aspect = float(k_len) / max(float(q_len), 1.0)
    base_panel_w = int(round(base_panel_h * aspect))
    panel_w = min(max(base_panel_w, 240), 1400)
    panel_h = max(base_panel_h, 96)
    margin_left = 48
    margin_top = 28
    gap = 4
    canvas_w = margin_left + len(block_ids) * panel_w + max(0, len(block_ids) - 1) * gap
    canvas_h = margin_top + len(steps) * panel_h + max(0, len(steps) - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = f"Flux2 Klein Attention Grid | step_stride={step_stride} block_stride={block_stride}"
    draw.text((margin_left, 6), title, fill=(0, 0, 0))

    for ridx, step in enumerate(steps):
        y0 = margin_top + ridx * (panel_h + gap)
        draw.text((4, y0 + 4), f"s{step}", fill=(0, 0, 0))
        cur_map = _load_step_block_entry(step_block_maps[step])
        for cidx, block_id in enumerate(block_ids):
            x0 = margin_left + cidx * (panel_w + gap)
            if ridx == 0:
                draw.text((x0 + 4, 6), f"b{block_id}", fill=(0, 0, 0))
            mat = cur_map.get(block_id)
            if mat is None:
                draw.rectangle([x0, y0, x0 + panel_w - 1, y0 + panel_h - 1], outline=(0, 0, 0))
                continue
            mat_np = mat.detach().float().cpu().numpy()
            rgb = render_segmented_attn_rgb(mat_np, norm_fn, k_spans)
            rgb = draw_span_boundaries_on_rgb(
                rgb=rgb,
                q_spans=q_spans,
                k_spans=k_spans,
                boundary_linewidth=float(boundary_linewidth),
            )
            tile = Image.fromarray(np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
            if tile.size != (panel_w, panel_h):
                tile = tile.resize((panel_w, panel_h), resample=_lanczos())
            canvas.paste(tile, (x0, y0))
            draw.rectangle([x0, y0, x0 + panel_w - 1, y0 + panel_h - 1], outline=(0, 0, 0), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format=str(save_format).upper())


def save_step_block_manifest(
    step_block_maps: Dict[int, Any],
    out_path: Path,
    key: str,
    prompt: str,
    q_ranges: List[Tuple[str, int, int]],
    k_ranges: List[Tuple[str, int, int]],
    q_sample_indices: List[int],
    k_sample_indices: List[int],
    block_info: Dict[int, Dict[str, Any]],
    aggregate_head: str,
    aggregate_block: str,
    step_stride: int,
    block_stride: int,
):
    steps, block_ids = _collect_step_and_block_ids(step_block_maps)
    sampled_block_info = filter_block_info(block_info, block_ids)
    manifest_files: Dict[str, str] = {}
    for step in steps:
        entry = step_block_maps[step]
        if isinstance(entry, (str, Path)):
            entry_path = Path(entry)
            try:
                rel = entry_path.relative_to(out_path.parent)
                manifest_files[str(step)] = str(rel)
            except Exception:
                manifest_files[str(step)] = str(entry_path)
        else:
            cache_name = f"_inline_step_{int(step):04d}.pt"
            cache_path = out_path.parent / cache_name
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(entry, cache_path)
            manifest_files[str(step)] = cache_name
    payload = {
        "storage_format": "step_block_cache_v1",
        "key": str(key),
        "prompt": str(prompt),
        "steps": [int(x) for x in steps],
        "block_ids": [int(x) for x in block_ids],
        "aggregate_head": str(aggregate_head),
        "aggregate_block": str(aggregate_block),
        "step_stride": int(step_stride),
        "block_stride": int(block_stride),
        "q_sample_indices": [int(x) for x in q_sample_indices],
        "k_sample_indices": [int(x) for x in k_sample_indices],
        "q_ranges": [(str(name), int(start), int(end)) for name, start, end in q_ranges],
        "k_ranges": [(str(name), int(start), int(end)) for name, start, end in k_ranges],
        "q_range_metadata": build_range_metadata(q_ranges, q_sample_indices),
        "k_range_metadata": build_range_metadata(k_ranges, k_sample_indices),
        "q_range_metadata_by_name": build_range_metadata_by_name(build_range_metadata(q_ranges, q_sample_indices)),
        "k_range_metadata_by_name": build_range_metadata_by_name(build_range_metadata(k_ranges, k_sample_indices)),
        "block_info": {int(k): dict(v) for k, v in sampled_block_info.items()},
        "step_block_files": manifest_files,
        "tensor_file_note": "Each step file stores a dict[int, float16 attention[q, k]] keyed by block_id.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def save_step_block_tensor(
    step_block_maps: Dict[int, Any],
    out_path: Path,
    key: str,
    prompt: str,
    q_ranges: List[Tuple[str, int, int]],
    k_ranges: List[Tuple[str, int, int]],
    q_sample_indices: List[int],
    k_sample_indices: List[int],
    block_info: Dict[int, Dict[str, Any]],
    aggregate_head: str,
    aggregate_block: str,
    step_stride: int,
    block_stride: int,
):
    tensor, mask, steps, block_ids = pack_step_block_maps(step_block_maps)
    sampled_block_info = filter_block_info(block_info, block_ids)
    payload = {
        "key": str(key),
        "prompt": str(prompt),
        "attn_tensor": tensor,
        "attn_mask": mask,
        "steps": [int(x) for x in steps],
        "block_ids": [int(x) for x in block_ids],
        "aggregate_head": str(aggregate_head),
        "aggregate_block": str(aggregate_block),
        "step_stride": int(step_stride),
        "block_stride": int(block_stride),
        "q_sample_indices": [int(x) for x in q_sample_indices],
        "k_sample_indices": [int(x) for x in k_sample_indices],
        "q_ranges": [(str(name), int(start), int(end)) for name, start, end in q_ranges],
        "k_ranges": [(str(name), int(start), int(end)) for name, start, end in k_ranges],
        "q_range_metadata": build_range_metadata(q_ranges, q_sample_indices),
        "k_range_metadata": build_range_metadata(k_ranges, k_sample_indices),
        "q_range_metadata_by_name": build_range_metadata_by_name(build_range_metadata(q_ranges, q_sample_indices)),
        "k_range_metadata_by_name": build_range_metadata_by_name(build_range_metadata(k_ranges, k_sample_indices)),
        "block_info": {int(k): dict(v) for k, v in sampled_block_info.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def save_token_layout_summary(
    out_path: Path,
    key: str,
    prompt: str,
    q_ranges: List[Tuple[str, int, int]],
    k_ranges: List[Tuple[str, int, int]],
    q_sample_indices: List[int],
    k_sample_indices: List[int],
    ref_infos: List[Dict[str, Any]],
    block_info: Dict[int, Dict[str, Any]],
):
    sampled_block_ids = sorted({int(k) for k in block_info.keys()})
    lines = [
        f"key={key}",
        f"prompt={prompt}",
        f"q_tokens_full={sum(max(int(e) - int(s), 0) for _, s, e in q_ranges)}",
        f"k_tokens_full={sum(max(int(e) - int(s), 0) for _, s, e in k_ranges)}",
        f"q_sample_indices={_summarize_indices(q_sample_indices, limit=20)}",
        f"k_sample_indices=[0-{len(k_sample_indices) - 1}] (n={len(k_sample_indices)})",
        f"q_ranges={q_ranges}",
        f"k_ranges={k_ranges}",
        f"ref_infos={json.dumps(ref_infos, ensure_ascii=False)}",
        f"sampled_block_ids={sampled_block_ids}",
        f"sampled_blocks={json.dumps(block_info, ensure_ascii=False)}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_prepare_ref_infos(pipe, images: List[Image.Image], ref_labels: List[str]) -> List[Dict[str, Any]]:
    ref_infos: List[Dict[str, Any]] = []
    if not images:
        return ref_infos
    for img, label in zip(images, ref_labels):
        image_width, image_height = img.size
        img_for_shape = img
        if image_width * image_height > 1024 * 1024:
            img_for_shape = pipe.image_processor._resize_to_target_area(img_for_shape, 1024 * 1024)
            image_width, image_height = img_for_shape.size
        multiple = pipe.vae_scale_factor * 2
        image_width, image_height = normalize_size_wh(image_width, image_height, multiple)
        ref_infos.append(
            {
                "label": str(label),
                "processed_size_wh": [int(image_width), int(image_height)],
                "token_count": compute_flux_token_count(image_width, image_height, pipe.vae_scale_factor),
            }
        )
    return ref_infos


def run_inference_worker(
    worker_idx: int,
    gpu: int,
    args,
    prompts: Dict[str, str],
    keys: List[str],
    key_to_index: Dict[str, int],
) -> Tuple[int, int, int]:
    use_cuda = torch.cuda.is_available() and gpu >= 0
    if use_cuda:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    dtype = choose_torch_dtype(device)
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(args.model_name, torch_dtype=dtype)
    if use_cuda and args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=gpu)
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=not bool(args.show_inner_progress))

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0
    text_encoder_out_layers = parse_text_encoder_out_layers(args.text_encoder_out_layers)

    for key in keys:
        collector: Optional[FluxAttentionCollector] = None
        step_cache_dir: Optional[Path] = None
        key_attn_dir: Optional[Path] = None
        out_img: Optional[Path] = None
        use_step_cache = False
        persist_step_cache = False
        try:
            if key not in prompts:
                print(f"[SKIP][worker={worker_idx}][gpu={gpu}] key不在prompts_json中: {key}")
                skipped += 1
                continue

            key_attn_dir = out_root / f"{key}_attn"
            out_img = out_root / f"{key}.png"
            if (not args.overwrite) and _has_complete_attention_outputs(
                key_attn_dir,
                out_img,
                args.save_format,
                save_attn_tensor=bool(args.save_attn_tensor),
            ):
                print(f"[SKIP][worker={worker_idx}][gpu={gpu}] 已存在完整输出: {key_attn_dir}")
                skipped += 1
                continue

            if args.overwrite:
                if key_attn_dir.exists():
                    shutil.rmtree(key_attn_dir, ignore_errors=True)
                if out_img.exists():
                    out_img.unlink()

            cref_path = Path(args.cref_dir) / f"{key}.png"
            sref_path = Path(args.sref_dir) / f"{key}.png"
            if not cref_path.exists() or not sref_path.exists():
                print(
                    f"[SKIP][worker={worker_idx}][gpu={gpu}] 图片缺失: "
                    f"key={key} cref={cref_path.exists()} sref={sref_path.exists()}"
                )
                skipped += 1
                continue

            prompt = prompts[key]
            cref = load_rgb(str(cref_path))
            sref = load_rgb(str(sref_path))
            orig_cref_size = cref.size
            orig_sref_size = sref.size
            if int(args.max_input_long_side) > 0:
                cref = resize_long_side_limit(cref, int(args.max_input_long_side))
                sref = resize_long_side_limit(sref, int(args.max_input_long_side))
                if cref.size != orig_cref_size or sref.size != orig_sref_size:
                    print(
                        f"[INPUT-RESIZE][worker={worker_idx}][gpu={gpu}] key={key} "
                        f"cref={orig_cref_size}->{cref.size} sref={orig_sref_size}->{sref.size}"
                    )

            if args.input_resolution:
                try:
                    w_str, h_str = args.input_resolution.lower().split("x", 1)
                    override_size = (int(w_str), int(h_str))
                except Exception as exc:
                    raise ValueError(f"invalid --input_resolution {args.input_resolution}") from exc
                cref = cref.resize(override_size, resample=_lanczos())
                content_size = override_size
            else:
                if int(args.max_input_long_side) > 0:
                    content_size = cref.size
                elif not args.no_resize_cref:
                    cref, content_size = resize_cref(cref)
                else:
                    content_size = cref.size

            images = [cref, sref]
            ref_labels = build_ref_labels(args.ref_labels, len(images))
            ref_infos = maybe_prepare_ref_infos(pipe, images, ref_labels)

            max_q_tokens = int(args.max_q_tokens) if int(args.max_q_tokens) > 0 else int(args.max_tokens)
            keep_full_query = bool(args.cache_full_attn)
            collector = FluxAttentionCollector(
                max_q_tokens=max_q_tokens,
                aggregate_head=str(args.aggregate_head),
                block_stride=int(args.block_stride),
                keep_full_query=keep_full_query,
                owner_label=f"[worker={worker_idx}][gpu={gpu}][key={key}]",
            )
            collector.register(pipe.transformer)

            multiple = pipe.vae_scale_factor * 2
            eff_w, eff_h = normalize_size_wh(content_size[0], content_size[1], multiple)
            latent_token_count = compute_flux_token_count(eff_w, eff_h, pipe.vae_scale_factor)
            token_ranges = build_joint_token_ranges(int(args.max_sequence_length), latent_token_count, ref_infos)
            seed_offset = int(key_to_index.get(key, 0))
            generator = torch.Generator(device=device).manual_seed(int(args.seed) + seed_offset)
            step_block_maps: Dict[int, Any] = {}
            use_step_cache = bool(args.cache_step_block) or bool(args.cache_full_attn)
            persist_step_cache = bool(args.cache_full_attn) and bool(args.save_attn_tensor)
            step_cache_dir = key_attn_dir / "_step_block_cache"

            def _store_step_block(step_i: int, cur: Dict[int, torch.Tensor]):
                if not cur:
                    return
                if use_step_cache:
                    step_cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = step_cache_dir / f"step_{int(step_i):04d}.pt"
                    torch.save(cur, cache_path)
                    step_block_maps[int(step_i)] = cache_path
                else:
                    step_block_maps[int(step_i)] = cur

            def on_step_end(_pipe, step, timestep, callback_kwargs):
                step_i = int(step)
                if step_i % max(int(args.step_stride), 1) != 0:
                    return callback_kwargs
                cur = {}
                for block_id in sorted(collector.maps.keys()):
                    cur[block_id] = collector.maps[block_id].clone()
                _store_step_block(step_i, cur)
                return callback_kwargs

            with torch.inference_mode():
                out = pipe(
                    image=images,
                    prompt=prompt,
                    width=content_size[0],
                    height=content_size[1],
                    num_inference_steps=int(args.steps),
                    guidance_scale=float(args.guidance_scale),
                    generator=generator,
                    callback_on_step_end=on_step_end,
                    max_sequence_length=int(args.max_sequence_length),
                    text_encoder_out_layers=text_encoder_out_layers,
                ).images[0]

            if collector.first_hook_error is not None:
                raise RuntimeError(
                    f"[worker={worker_idx}][gpu={gpu}] attention hook失败: key={key}; "
                    f"hook_errors={collector.hook_error_count}; "
                    f"first_hook_error={collector.describe_first_hook_error()}"
                )

            out.save(out_img)

            if not step_block_maps:
                cur = {}
                for block_id in sorted(collector.maps.keys()):
                    cur[block_id] = collector.maps[block_id].clone()
                _store_step_block(0, cur)
            if not step_block_maps:
                detail = collector.describe_first_hook_error()
                if detail:
                    raise RuntimeError(
                        f"[worker={worker_idx}][gpu={gpu}] 没有收集到attention: key={key}; "
                        f"hook_errors={collector.hook_error_count}; first_hook_error={detail}"
                    )
                raise RuntimeError(f"[worker={worker_idx}][gpu={gpu}] 没有收集到attention: key={key}")

            q_sample_indices = [int(x) for x in collector.shared_meta.get("q_sample_indices", [])]
            k_sample_indices = [int(x) for x in collector.shared_meta.get("k_sample_indices", [])]
            if not q_sample_indices:
                q_sample_indices = list(range(sum(max(int(e) - int(s), 0) for _, s, e in token_ranges)))
            if not k_sample_indices:
                k_sample_indices = list(range(sum(max(int(e) - int(s), 0) for _, s, e in token_ranges)))

            _, sampled_block_ids = _collect_step_and_block_ids(step_block_maps)
            sampled_block_info = filter_block_info(collector.block_info, sampled_block_ids)

            out_map = key_attn_dir / f"attention_step_block_grid.{args.save_format}"
            save_step_block_grid(
                step_block_maps=step_block_maps,
                out_path=out_map,
                q_ranges=token_ranges,
                k_ranges=token_ranges,
                q_sample_indices=q_sample_indices,
                k_sample_indices=k_sample_indices,
                image_dpi=int(args.image_dpi),
                save_format=str(args.save_format),
                step_stride=max(int(args.step_stride), 1),
                block_stride=max(int(args.block_stride), 1),
                panel_size=float(args.panel_size),
                attn_gamma=float(args.attn_gamma),
                boundary_linewidth=float(args.boundary_linewidth),
            )

            summary_path = key_attn_dir / "attention_region_summary.json"
            save_region_summary(
                step_block_maps=step_block_maps,
                out_path=summary_path,
                q_ranges=token_ranges,
                k_ranges=token_ranges,
                q_sample_indices=q_sample_indices,
                k_sample_indices=k_sample_indices,
                block_info=sampled_block_info,
                aggregate_block=str(args.aggregate_block),
            )

            token_summary = key_attn_dir / "token_layout_summary.txt"
            save_token_layout_summary(
                out_path=token_summary,
                key=key,
                prompt=prompt,
                q_ranges=token_ranges,
                k_ranges=token_ranges,
                q_sample_indices=q_sample_indices,
                k_sample_indices=k_sample_indices,
                ref_infos=ref_infos,
                block_info=sampled_block_info,
            )

            if args.save_attn_tensor:
                out_tensor = key_attn_dir / "attention_step_block_grid.pt"
                if args.cache_full_attn:
                    save_step_block_manifest(
                        step_block_maps=step_block_maps,
                        out_path=out_tensor,
                        key=key,
                        prompt=prompt,
                        q_ranges=token_ranges,
                        k_ranges=token_ranges,
                        q_sample_indices=q_sample_indices,
                        k_sample_indices=k_sample_indices,
                        block_info=sampled_block_info,
                        aggregate_head=str(args.aggregate_head),
                        aggregate_block=str(args.aggregate_block),
                        step_stride=max(int(args.step_stride), 1),
                        block_stride=max(int(args.block_stride), 1),
                    )
                else:
                    save_step_block_tensor(
                        step_block_maps=step_block_maps,
                        out_path=out_tensor,
                        key=key,
                        prompt=prompt,
                        q_ranges=token_ranges,
                        k_ranges=token_ranges,
                        q_sample_indices=q_sample_indices,
                        k_sample_indices=k_sample_indices,
                        block_info=sampled_block_info,
                        aggregate_head=str(args.aggregate_head),
                        aggregate_block=str(args.aggregate_block),
                        step_stride=max(int(args.step_stride), 1),
                        block_stride=max(int(args.block_stride), 1),
                    )

            done += 1
            print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} generated_image={out_img}")
            print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} fullmap={out_map}")
            if args.save_attn_tensor:
                print(
                    f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} "
                    f"attn_tensor={key_attn_dir / 'attention_step_block_grid.pt'}"
                )
                if args.cache_full_attn:
                    print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} attn_step_cache={step_cache_dir}")
            print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} token_layout={token_summary}")
            print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} region_summary={summary_path}")
            print(f"[SELF-CHECK][worker={worker_idx}][gpu={gpu}] key={key} {'PASS' if out_map.exists() else 'FAIL'}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL][worker={worker_idx}][gpu={gpu}] key={key} {type(exc).__name__}: {exc}", flush=True)
            if key_attn_dir is not None and key_attn_dir.exists():
                shutil.rmtree(key_attn_dir, ignore_errors=True)
            if out_img is not None and out_img.exists():
                out_img.unlink()
        finally:
            if collector is not None:
                try:
                    collector.remove()
                except Exception:
                    pass
            if step_cache_dir is not None and use_step_cache and step_cache_dir.exists() and not persist_step_cache:
                shutil.rmtree(step_cache_dir, ignore_errors=True)
            if use_cuda:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    return done, skipped, failed


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_json)
    keys = read_keys(args.key_txt)
    key_to_index = {k: i for i, k in enumerate(keys)}
    gpu_list = parse_gpu_list(args.gpus)
    if torch.cuda.is_available():
        gpu_count = int(torch.cuda.device_count())
        invalid = [g for g in gpu_list if g >= gpu_count]
        if invalid:
            raise RuntimeError(f"GPU编号越界: {invalid}，当前可见GPU数量={gpu_count}")
    else:
        gpu_list = [0]

    if len(gpu_list) <= 1 or not torch.cuda.is_available():
        done, skipped, failed = run_inference_worker(
            worker_idx=0,
            gpu=int(gpu_list[0]),
            args=args,
            prompts=prompts,
            keys=keys,
            key_to_index=key_to_index,
        )
    else:
        chunks = split_keys_round_robin(keys, len(gpu_list))
        tasks = []
        for worker_idx, (gpu, sub_keys) in enumerate(zip(gpu_list, chunks)):
            if not sub_keys:
                continue
            tasks.append((worker_idx, int(gpu), args, prompts, sub_keys, key_to_index))
        mp_ctx = mp.get_context("spawn")
        with mp_ctx.Pool(processes=len(tasks)) as pool:
            results = [pool.apply_async(run_inference_worker, args=task) for task in tasks]
            results = [res.get() for res in results]
        done = int(sum(x[0] for x in results))
        skipped = int(sum(x[1] for x in results))
        failed = int(sum(x[2] for x in results))

    print(f"[FINAL] done={done} skipped={skipped} failed={failed} total={len(keys)}")
    if failed > 0:
        raise RuntimeError(f"[FINAL] 存在失败key: failed={failed} total={len(keys)}")


if __name__ == "__main__":
    main()

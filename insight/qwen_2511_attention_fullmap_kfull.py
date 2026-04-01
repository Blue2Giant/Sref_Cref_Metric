#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import multiprocessing as mp
from queue import Empty
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Qwen-Image-Edit-2511 Attention FullMap 可视化（K全量，Q采样）")
    parser.add_argument("--prompts_json", required=True, help="id->prompt 的json文件")
    parser.add_argument("--cref_dir", required=True, help="内容参考图目录，文件名应为 {id}.png")
    parser.add_argument("--sref_dir", required=True, help="风格参考图目录，文件名应为 {id}.png")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--gpus", default="0", help='如 "0" 或 "0,1,2"，会在这些GPU上并行推理')
    parser.add_argument("--key_txt", required=True, help="txt文件，可包含多行key")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128, help="attention矩阵q/k最大采样token数")
    parser.add_argument("--max-q-tokens", type=int, default=0, help="query采样token上限，<=0时使用max-tokens")
    parser.add_argument("--max-k-tokens", type=int, default=0, help="兼容保留参数；本脚本在K维不采样，会保留全部key token")
    parser.add_argument("--aggregate-head", choices=["mean", "max"], default="mean")
    parser.add_argument("--aggregate-block", choices=["mean", "max"], default="mean")
    parser.add_argument("--step-stride", type=int, default=4, help="每隔多少步采样一个step")
    parser.add_argument("--block-stride", type=int, default=4, help="每隔多少个block采样一个")
    parser.add_argument("--panel-size", type=float, default=1.4, help="step-block拼图每个子图尺寸")
    parser.add_argument("--image-dpi", type=int, default=120)
    parser.add_argument("--save-format", choices=["png", "pdf"], default="png")
    parser.add_argument("--ref-labels", default="cref,sref", help="参考图名称，逗号分隔，按输入图顺序")
    parser.add_argument("--attn-cmap", default="turbo", help="attention强度色图")
    parser.add_argument("--attn-gamma", type=float, default=0.5, help="attention强度gamma增强，越小越强调高值")
    parser.add_argument("--high-attn-quantile", type=float, default=0.9, help="高注意力区域分位数阈值")
    parser.add_argument("--high-attn-contour-color", default="#00e5ff", help="高注意力区域轮廓线颜色")
    parser.add_argument("--high-attn-contour-width", type=float, default=0.8, help="高注意力区域轮廓线宽")
    parser.add_argument("--region-alpha", type=float, default=0.18, help="参考图分区底色透明度")
    parser.add_argument("--boundary-linewidth", type=float, default=2.0, help="参考图分界线宽度")
    parser.add_argument("--save-attn-tensor", action="store_true", help="额外保存用于绘图的attention张量为pt文件")
    parser.add_argument("--show-inner-progress", action="store_true", help="显示diffusers内部step进度条；默认关闭，仅显示总进度")
    parser.add_argument("--overwrite", action="store_true")
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


def _dir_has_any_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for _ in path.iterdir():
        return True
    return False


def _is_valid_output_file(path: Path, save_format: str) -> bool:
    if not path.exists() or (not path.is_file()):
        return False
    try:
        size = int(path.stat().st_size)
    except Exception:
        return False
    if size <= 65536:
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


def _is_nonempty_file(path: Path) -> bool:
    if not path.exists() or (not path.is_file()):
        return False
    try:
        return int(path.stat().st_size) > 0
    except Exception:
        return False


def _has_complete_attention_outputs(
    key_attn_dir: Path,
    out_img: Path,
    save_format: str,
    save_attn_tensor: bool = False,
) -> bool:
    out_map = key_attn_dir / f"attention_step_block_grid.{save_format}"
    if not _is_valid_output_file(out_map, save_format):
        return False
    if not _is_nonempty_file(out_img):
        return False
    if bool(save_attn_tensor):
        out_tensor = key_attn_dir / "attention_step_block_grid.pt"
        if not _is_nonempty_file(out_tensor):
            return False
    return True


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


class AttentionCollector:
    def __init__(self, max_q_tokens: int, max_k_tokens: int):
        self.max_q_tokens = max(1, int(max_q_tokens))
        self.max_k_tokens = max(1, int(max_k_tokens))
        self.handles = []
        self.maps: Dict[int, torch.Tensor] = {}
        self.meta: Dict[int, Dict[str, object]] = {}

    def _sample_tokens(self, n: int, limit: int) -> torch.Tensor:
        if n <= limit:
            return torch.arange(n, dtype=torch.long)
        return torch.linspace(0, n - 1, steps=limit).long()

    def _keep_all_tokens(self, n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.long)

    def _hook_fn(self, block_idx: int):
        def _fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            if hidden_states is None or not torch.is_tensor(hidden_states):
                return
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            key_states = encoder_hidden_states if torch.is_tensor(encoder_hidden_states) else hidden_states
            if not hasattr(module, "to_q") or not hasattr(module, "to_k") or not hasattr(module, "heads"):
                return
            try:
                q = module.to_q(hidden_states)
                k = module.to_k(key_states)
                b, qn, qc = q.shape
                _bk, kn, kc = k.shape
                heads = int(module.heads)
                if heads <= 0 or qc % heads != 0 or kc % heads != 0:
                    return
                hd = qc // heads
                q = q.view(b, qn, heads, hd).permute(0, 2, 1, 3)
                k = k.view(b, kn, heads, hd).permute(0, 2, 1, 3)
                scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(hd))
                attn = torch.softmax(scores, dim=-1)
                attn = attn[0].detach().float().cpu()
                q_idx = self._sample_tokens(attn.shape[1], self.max_q_tokens)
                k_idx = self._keep_all_tokens(attn.shape[2])
                attn = attn[:, q_idx][:, :, k_idx]
                self.maps[block_idx] = attn
                self.meta[block_idx] = {
                    "has_encoder": bool(torch.is_tensor(encoder_hidden_states)),
                    "q_tokens_full": int(qn),
                    "k_tokens_full": int(kn),
                    "q_sample_indices": q_idx.tolist(),
                    "k_sample_indices": k_idx.tolist(),
                }
            except Exception:
                return
        return _fn

    def register(self, model: torch.nn.Module):
        idx = 0
        for _, module in model.named_modules():
            has_core = hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "heads")
            if not has_core:
                continue
            h = module.register_forward_hook(self._hook_fn(idx), with_kwargs=True)
            self.handles.append(h)
            idx += 1

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_ref_ranges(total_len: int, num_refs: int) -> List[Tuple[int, int, int]]:
    if num_refs <= 0:
        return []
    base = total_len // num_refs
    rem = total_len % num_refs
    out = []
    start = 0
    for i in range(num_refs):
        span = base + (1 if i < rem else 0)
        end = start + span
        out.append((i, start, end))
        start = end
    return out


def build_ref_ticks(ranges: List[Tuple[int, int, int]], labels: List[str], prefix: str) -> Tuple[List[float], List[str]]:
    ticks: List[float] = []
    texts: List[str] = []
    for ridx, start, end in ranges:
        if end <= start or ridx >= len(labels):
            continue
        ticks.append((start + end - 1) * 0.5)
        texts.append(f"{prefix}:{labels[ridx]} [{start}-{end - 1}]")
    return ticks, texts


def build_axis_layout(
    full_len: int,
    sample_indices: List[int],
    labels: List[str],
    prefix: str,
) -> Tuple[List[float], List[str], List[int]]:
    num_refs = len(labels)
    if num_refs <= 0:
        return [], [], []
    n_sample = len(sample_indices)
    if n_sample <= 0:
        return [], [], []
    full_ranges = build_ref_ranges(max(int(full_len), 1), num_refs)
    arr = np.asarray(sample_indices, dtype=np.int64)
    sample_ranges: List[Tuple[int, int, int, int, int]] = []
    for ridx, fs, fe in full_ranges:
        if fe <= fs:
            continue
        loc = np.where((arr >= fs) & (arr < fe))[0]
        if loc.size <= 0:
            continue
        ss = int(loc.min())
        se = int(loc.max()) + 1
        sample_ranges.append((ridx, ss, se, fs, fe))
    if not sample_ranges:
        fallback = build_ref_ranges(n_sample, num_refs)
        sample_ranges = [(ridx, ss, se, ss, se) for ridx, ss, se in fallback if se > ss]
    ticks: List[float] = []
    tick_labels: List[str] = []
    boundaries: List[int] = []
    for i, (ridx, ss, se, fs, fe) in enumerate(sample_ranges):
        if ridx >= len(labels) or se <= ss:
            continue
        ticks.append((ss + se - 1) * 0.5)
        tick_labels.append(f"{prefix}:{labels[ridx]} [{fs}-{fe - 1}]")
        if i < len(sample_ranges) - 1:
            boundaries.append(se)
    return ticks, tick_labels, boundaries


def build_k_semantic_ranges(k_full: int, has_encoder: bool, text_tokens_est: int, ref_labels: List[str]) -> List[Tuple[str, int, int]]:
    k_full = max(0, int(k_full))
    if k_full <= 0:
        return []
    if not has_encoder:
        return [("noise", 0, k_full)]
    text_len = min(max(int(text_tokens_est), 0), k_full)
    ranges: List[Tuple[str, int, int]] = []
    cursor = 0
    if text_len > 0:
        ranges.append(("text", 0, text_len))
        cursor = text_len
    remain = max(0, k_full - cursor)
    if remain > 0 and len(ref_labels) > 0:
        ref_ranges = build_ref_ranges(remain, len(ref_labels))
        for ridx, rs, re in ref_ranges:
            name = str(ref_labels[ridx]) if ridx < len(ref_labels) else f"ref{ridx}"
            ranges.append((name, cursor + rs, cursor + re))
        cursor = k_full
    if cursor < k_full:
        ranges.append(("noise", cursor, k_full))
    return [(name, s, e) for name, s, e in ranges if e > s]


def map_ranges_to_sample_ticks(
    ranges: List[Tuple[str, int, int]],
    sample_indices: List[int],
    prefix: str,
) -> Tuple[List[float], List[str], List[int]]:
    if not ranges or not sample_indices:
        return [], [], []
    arr = np.asarray(sample_indices, dtype=np.int64)
    ticks: List[float] = []
    labels: List[str] = []
    boundaries: List[int] = []
    valid_count = 0
    mapped: List[Tuple[str, int, int, int, int]] = []
    for name, fs, fe in ranges:
        loc = np.where((arr >= int(fs)) & (arr < int(fe)))[0]
        if loc.size <= 0:
            continue
        ss = int(loc.min())
        se = int(loc.max()) + 1
        mapped.append((name, ss, se, int(fs), int(fe)))
    for idx, (name, ss, se, fs, fe) in enumerate(mapped):
        if se <= ss:
            continue
        valid_count += 1
        ticks.append((ss + se - 1) * 0.5)
        labels.append(f"{prefix}:{name} [{fs}-{fe - 1}]")
        if idx < len(mapped) - 1:
            boundaries.append(se)
    if valid_count <= 0:
        return [], [], []
    return ticks, labels, boundaries


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
            mapped.append((name, ss, se))
    return mapped


def build_range_metadata(
    ranges: List[Tuple[str, int, int]],
    sample_indices: List[int],
) -> List[Dict[str, object]]:
    arr = np.asarray(sample_indices or [], dtype=np.int64)
    out: List[Dict[str, object]] = []
    for name, fs, fe in ranges:
        item: Dict[str, object] = {
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


def build_range_metadata_by_name(items: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for item in items:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        out[name] = dict(item)
    return out


def _k_base_color(name: str) -> np.ndarray:
    s = str(name or "").lower()
    if "cref" in s:
        return np.array([0.35, 0.05, 0.05], dtype=np.float32)
    if "sref" in s:
        return np.array([0.04, 0.10, 0.35], dtype=np.float32)
    return np.array([0.05, 0.22, 0.10], dtype=np.float32)


def render_segmented_attn_rgb(
    mat_np: np.ndarray,
    norm_obj,
    k_spans: List[Tuple[str, int, int]],
) -> np.ndarray:
    h, w = mat_np.shape
    base_cols = np.tile(np.array([0.05, 0.22, 0.10], dtype=np.float32), (w, 1))
    for name, s, e in k_spans:
        ss = max(0, int(s))
        ee = min(w, int(e))
        if ee > ss:
            base_cols[ss:ee, :] = _k_base_color(name)
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    v = np.asarray(norm_obj(mat_np), dtype=np.float32)
    v = np.clip(v, 0.0, 1.0)[..., None]
    rgb = base_cols[None, :, :] * (1.0 - v) + yellow[None, None, :] * v
    return np.clip(rgb, 0.0, 1.0)


def aggregate_full_map(attn_maps: Dict[int, torch.Tensor], head_mode: str, block_mode: str) -> torch.Tensor:
    if len(attn_maps) == 0:
        raise RuntimeError("没有收集到attention map")
    blocks = sorted(attn_maps.keys())
    q_target = max(int(attn_maps[b].shape[1]) for b in blocks)
    k_target = max(int(attn_maps[b].shape[2]) for b in blocks)
    mats = []
    for b in blocks:
        x = attn_maps[b]
        if head_mode == "max":
            x = x.max(dim=0).values
        else:
            x = x.mean(dim=0)
        if int(x.shape[0]) != q_target or int(x.shape[1]) != k_target:
            x = F.interpolate(
                x.unsqueeze(0).unsqueeze(0),
                size=(q_target, k_target),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        mats.append(x)
    stack = torch.stack(mats, dim=0)
    if block_mode == "max":
        return stack.max(dim=0).values
    return stack.mean(dim=0)


def aggregate_head_map(attn: torch.Tensor, head_mode: str) -> torch.Tensor:
    if head_mode == "max":
        return attn.max(dim=0).values
    return attn.mean(dim=0)


def _load_step_block_entry(entry: Any) -> Dict[int, torch.Tensor]:
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, (str, Path)):
        data = torch.load(entry, map_location="cpu")
        if isinstance(data, dict):
            return data
    raise TypeError(f"Unsupported step-block entry type: {type(entry)}")


def _collect_step_and_block_ids(step_block_maps: Dict[int, Any]) -> Tuple[List[int], List[int]]:
    if len(step_block_maps) == 0:
        raise RuntimeError("没有可用的step-block attention")
    steps = sorted(int(s) for s in step_block_maps.keys())
    block_ids = set()
    for step in steps:
        cur = _load_step_block_entry(step_block_maps[step])
        for block_id in cur.keys():
            block_ids.add(int(block_id))
    block_ids = sorted(block_ids)
    if len(block_ids) == 0:
        raise RuntimeError("没有可用的block attention")
    return steps, block_ids


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
        cur = _load_step_block_entry(step_block_maps[step])
        total_entries += max(1, len(cur))

    samples: List[torch.Tensor] = []
    global_min = None
    global_max = None
    remaining_budget = max(1, int(max_points))
    remaining_entries = max(1, int(total_entries))
    for step in steps:
        cur = _load_step_block_entry(step_block_maps[step])
        for block_id in block_ids:
            mat = cur.get(block_id)
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
        raise RuntimeError("step-block attention 为空")

    flat = torch.cat(samples, dim=0)
    return _estimate_quantiles_from_flat_sample(
        flat,
        q_low=q_low,
        q_high=q_high,
        fallback_min=global_min,
        fallback_max=global_max,
    )


def _estimate_value_range_from_tensor(
    tensor: torch.Tensor,
    q_low: float = 0.02,
    q_high: float = 0.995,
    max_points: int = 1_000_000,
) -> Tuple[float, float]:
    flat = _sample_flat_tensor(tensor, max_points=max_points)
    return _estimate_quantiles_from_flat_sample(
        flat,
        q_low=q_low,
        q_high=q_high,
        fallback_min=float(tensor.min().item()),
        fallback_max=float(tensor.max().item()),
    )


def pack_step_block_maps(step_block_maps: Dict[int, Any]) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    if len(step_block_maps) == 0:
        raise RuntimeError("没有可用的step-block attention")
    steps, block_ids = _collect_step_and_block_ids(step_block_maps)

    q_target = 0
    k_target = 0
    for step in steps:
        cur = _load_step_block_entry(step_block_maps[step])
        for block_id in block_ids:
            mat = cur.get(block_id)
            if mat is None:
                continue
            q_target = max(q_target, int(mat.shape[0]))
            k_target = max(k_target, int(mat.shape[1]))
    if q_target <= 0 or k_target <= 0:
        raise RuntimeError("step-block attention 为空")

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
                cur = F.interpolate(
                    cur.unsqueeze(0).unsqueeze(0),
                    size=(q_target, k_target),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            tensor[sidx, bidx] = cur
            mask[sidx, bidx] = True
    return tensor, mask, steps, block_ids


def save_step_block_tensor(
    step_block_maps: Dict[int, Dict[int, torch.Tensor]],
    out_path: Path,
    key: str,
    prompt: str,
    ref_labels: List[str],
    aggregate_head: str,
    step_stride: int,
    block_stride: int,
    q_tokens_full: int = 0,
    k_tokens_full: int = 0,
    q_sample_indices: List[int] = None,
    k_sample_indices: List[int] = None,
    has_encoder: bool = False,
    text_tokens_est: int = 0,
    collector_meta: Dict[int, Dict[str, object]] = None,
):
    tensor, mask, steps, block_ids = pack_step_block_maps(step_block_maps)
    q_tokens_full = int(q_tokens_full) if int(q_tokens_full) > 0 else int(tensor.shape[-2])
    k_tokens_full = int(k_tokens_full) if int(k_tokens_full) > 0 else int(tensor.shape[-1])
    q_sample_indices = [int(x) for x in (q_sample_indices or list(range(int(tensor.shape[-2]))))]
    k_sample_indices = [int(x) for x in (k_sample_indices or [])]
    if not k_sample_indices:
        k_sample_indices = list(range(int(tensor.shape[-1])))
    k_semantic_ranges = build_k_semantic_ranges(k_tokens_full, bool(has_encoder), int(text_tokens_est), ref_labels)
    k_range_metadata = build_range_metadata(k_semantic_ranges, k_sample_indices)
    payload = {
        "key": str(key),
        "prompt": str(prompt),
        "attn_tensor": tensor,
        "attn_mask": mask,
        "steps": [int(x) for x in steps],
        "block_ids": [int(x) for x in block_ids],
        "ref_labels": [str(x) for x in ref_labels],
        "aggregate_head": str(aggregate_head),
        "step_stride": int(step_stride),
        "block_stride": int(block_stride),
        "q_tokens_full": q_tokens_full,
        "k_tokens_full": k_tokens_full,
        "q_sample_indices": q_sample_indices,
        "k_sample_indices": k_sample_indices,
        "has_encoder": bool(has_encoder),
        "text_tokens_est": int(text_tokens_est),
        "k_semantic_ranges": [(str(name), int(start), int(end)) for name, start, end in k_semantic_ranges],
        "k_range_metadata": k_range_metadata,
        "k_range_metadata_by_name": build_range_metadata_by_name(k_range_metadata),
        "collector_meta": collector_meta or {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def build_high_mask_overlay(arr: np.ndarray, thr: float, color_hex: str, alpha: float) -> np.ndarray:
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    mask = arr >= float(thr)
    if not np.any(mask):
        return rgba
    rgb = np.array(mcolors.to_rgb(color_hex), dtype=np.float32)
    rgba[mask, 0] = rgb[0]
    rgba[mask, 1] = rgb[1]
    rgba[mask, 2] = rgb[2]
    rgba[mask, 3] = float(alpha)
    return rgba


def save_token_layout_summary(
    out_path: Path,
    key: str,
    prompt: str,
    ref_labels: List[str],
    collector: AttentionCollector,
    sample_block: int,
    step_block_maps: Dict[int, Dict[int, torch.Tensor]],
    tokenizer=None,
):
    meta = collector.meta.get(sample_block, {})
    q_full = int(meta.get("q_tokens_full", 0))
    k_full = int(meta.get("k_tokens_full", 0))
    has_encoder = bool(meta.get("has_encoder", False))
    q_sample = [int(x) for x in meta.get("q_sample_indices", [])]
    k_sample = [int(x) for x in meta.get("k_sample_indices", [])]
    q_text = "[]"
    k_text = "[]"
    q_non_text = "[]"
    k_non_text = "[]"
    text_tokens_est = 0
    if tokenizer is not None:
        try:
            text_tokens_est = int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
        except Exception:
            text_tokens_est = 0
    if q_full > 0:
        q_non_text = f"[0-{q_full - 1}]"
    if has_encoder and k_full > 0:
        text_end = min(max(text_tokens_est - 1, -1), k_full - 1)
        if text_end >= 0:
            k_text = f"[0-{text_end}]"
            if text_end + 1 <= k_full - 1:
                k_non_text = f"[{text_end + 1}-{k_full - 1}]"
        else:
            k_non_text = f"[0-{k_full - 1}]"
    elif k_full > 0:
        k_non_text = f"[0-{k_full - 1}]"
    lines = [
        f"key={key}",
        f"sample_block={sample_block}",
        f"head_aggregation=mean_or_max_by_arg",
        f"q_source=hidden_states(latent/noise)",
        f"k_source={'encoder_hidden_states(cond: text+image/ref)' if has_encoder else 'hidden_states(latent/noise)'}",
        f"q_tokens_full={q_full}",
        f"k_tokens_full={k_full}",
        f"text_tokens_estimated={text_tokens_est}",
        f"q_text_range={q_text}",
        f"q_non_text_range={q_non_text}",
        f"k_text_range_estimated={k_text}",
        f"k_non_text_range_estimated={k_non_text}",
        f"q_sampled_indices={q_sample}",
        f"k_sampled_indices={k_sample}",
        f"ref_labels={ref_labels}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_step_block_grid(
    step_block_maps: Dict[int, Any],
    out_path: Path,
    ref_labels: List[str],
    image_dpi: int,
    save_format: str,
    head_mode: str,
    step_stride: int,
    block_stride: int,
    panel_size: float,
    attn_cmap: str,
    attn_gamma: float,
    high_attn_quantile: float,
    high_attn_contour_color: str,
    high_attn_contour_width: float,
    region_alpha: float,
    boundary_linewidth: float,
    q_tokens_full: int = 0,
    k_tokens_full: int = 0,
    q_sample_indices: List[int] = None,
    k_sample_indices: List[int] = None,
    has_encoder: bool = False,
    text_tokens_est: int = 0,
):
    if len(step_block_maps) == 0:
        raise RuntimeError("没有可用的step-block attention")
    steps, block_ids = _collect_step_and_block_ids(step_block_maps)

    sample_map = _load_step_block_entry(step_block_maps[steps[0]])
    sample = sample_map[block_ids[0]]
    q_indices = [int(x) for x in (q_sample_indices or list(range(int(sample.shape[0]))))]
    k_indices = [int(x) for x in (k_sample_indices or list(range(int(sample.shape[1]))))]
    q_full = int(q_tokens_full) if int(q_tokens_full) > 0 else int(sample.shape[0])
    k_full = int(k_tokens_full) if int(k_tokens_full) > 0 else int(sample.shape[1])
    k_semantic_ranges = build_k_semantic_ranges(k_full, bool(has_encoder), int(text_tokens_est), ref_labels)
    k_spans = map_ranges_to_sample_spans(k_semantic_ranges, k_indices)
    vmin, vmax = _estimate_value_range_from_step_block_maps(step_block_maps)
    norm = mcolors.PowerNorm(gamma=max(float(attn_gamma), 1e-3), vmin=vmin, vmax=vmax)
    panel_size = max(0.8, float(panel_size))
    fig_w = max(10.0, len(block_ids) * panel_size)
    fig_h = max(8.0, len(steps) * panel_size)
    fig, axes = plt.subplots(len(steps), len(block_ids), figsize=(fig_w, fig_h), squeeze=False)
    for ridx, step in enumerate(steps):
        cur_map = _load_step_block_entry(step_block_maps[step])
        for cidx, block_id in enumerate(block_ids):
            ax = axes[ridx][cidx]
            mat = cur_map.get(block_id)
            if mat is None:
                ax.axis("off")
                continue
            mat_np = mat.numpy()
            rgb = render_segmented_attn_rgb(mat_np, norm, k_spans)
            ax.imshow(rgb, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color("black")
                sp.set_linewidth(0.8)
    fig.suptitle(
        f"Attention Grid (rows=step stride {step_stride}, cols=block stride {block_stride}, head={head_mode})",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.03, right=0.997, top=0.96, bottom=0.08, wspace=0.0, hspace=0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=image_dpi, bbox_inches="tight", format=save_format)
    plt.close(fig)


def save_full_map(
    full_map: torch.Tensor,
    out_path: Path,
    ref_labels: List[str],
    image_dpi: int,
    save_format: str,
    head_mode: str,
    block_mode: str,
    attn_cmap: str,
    attn_gamma: float,
    high_attn_quantile: float,
    high_attn_contour_color: str,
    high_attn_contour_width: float,
    region_alpha: float,
    boundary_linewidth: float,
    has_encoder: bool = False,
    text_tokens_est: int = 0,
):
    q_len = int(full_map.shape[0])
    k_len = int(full_map.shape[1])
    k_indices = list(range(k_len))
    k_semantic_ranges = build_k_semantic_ranges(k_len, bool(has_encoder), int(text_tokens_est), ref_labels)
    k_spans = map_ranges_to_sample_spans(k_semantic_ranges, k_indices)
    vmin, vmax = _estimate_value_range_from_tensor(full_map)
    norm = mcolors.PowerNorm(gamma=max(float(attn_gamma), 1e-3), vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    full_np = full_map.numpy()
    rgb = render_segmented_attn_rgb(full_np, norm, k_spans)
    ax.imshow(rgb, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Full Attention Map (head={head_mode}, block={block_mode})")
    ax.set_xlabel("key tokens")
    ax.set_ylabel("query tokens")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=image_dpi, bbox_inches="tight", format=save_format)
    plt.close(fig)


def _emit_progress(
    progress_queue,
    status: str,
    key: str,
    worker_idx: int,
    gpu: int,
):
    if progress_queue is None:
        return
    try:
        progress_queue.put(
            {
                "status": str(status),
                "key": str(key),
                "worker_idx": int(worker_idx),
                "gpu": int(gpu),
            }
        )
    except Exception:
        return


def run_inference_worker(
    worker_idx: int,
    gpu: int,
    args,
    prompts: Dict[str, str],
    keys: List[str],
    key_to_index: Dict[str, int],
    progress_queue=None,
    show_local_progress: bool = False,
) -> Tuple[int, int, int]:
    use_cuda = torch.cuda.is_available() and gpu >= 0
    if use_cuda:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=not bool(args.show_inner_progress))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0
    pbar: Optional[tqdm] = None
    if show_local_progress:
        pbar = tqdm(total=len(keys), desc=f"total gpu{gpu}", dynamic_ncols=True)

    for key in keys:
        status = "failed"
        collector = None
        step_cache_dir: Optional[Path] = None
        try:
            if key not in prompts:
                print(f"[SKIP][worker={worker_idx}][gpu={gpu}] key不在prompts_json中，已跳过: {key}")
                skipped += 1
                status = "skipped"
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
                status = "skipped"
                continue
            if args.overwrite:
                if key_attn_dir.exists():
                    shutil.rmtree(key_attn_dir, ignore_errors=True)
                if out_img.exists():
                    out_img.unlink()
            cref_path = Path(args.cref_dir) / f"{key}.png"
            sref_path = Path(args.sref_dir) / f"{key}.png"
            if not cref_path.exists() or not sref_path.exists():
                print(f"[SKIP][worker={worker_idx}][gpu={gpu}] 图片缺失: key={key} cref={cref_path.exists()} sref={sref_path.exists()}")
                skipped += 1
                status = "skipped"
                continue

            prompt = prompts[key]
            cref = load_rgb(str(cref_path))
            sref = load_rgb(str(sref_path))
            images = [cref, sref]
            ref_labels = [x.strip() for x in str(args.ref_labels).split(",") if x.strip()]
            if len(ref_labels) < len(images):
                ref_labels += [f"ref{i}" for i in range(len(ref_labels), len(images))]
            ref_labels = ref_labels[: len(images)]

            max_q_tokens = int(args.max_q_tokens) if int(args.max_q_tokens) > 0 else int(args.max_tokens)
            max_k_tokens = int(args.max_k_tokens) if int(args.max_k_tokens) > 0 else int(args.max_tokens)
            collector = AttentionCollector(max_q_tokens=max_q_tokens, max_k_tokens=max_k_tokens)
            text_tokens_est = 0
            if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
                try:
                    text_tokens_est = int(pipe.tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
                except Exception:
                    text_tokens_est = 0
            if hasattr(pipe, "transformer") and pipe.transformer is not None:
                collector.register(pipe.transformer)
            else:
                collector.register(pipe.unet if hasattr(pipe, "unet") else pipe)

            seed_offset = int(key_to_index.get(key, 0))
            generator = torch.Generator(device=device).manual_seed(int(args.seed) + seed_offset)
            step_block_maps: Dict[int, Any] = {}
            step_cache_dir = key_attn_dir / "_step_block_cache"
            if step_cache_dir.exists():
                shutil.rmtree(step_cache_dir, ignore_errors=True)

            def on_step_end(_pipe, step, timestep, callback_kwargs):
                step_i = int(step)
                if step_i % max(int(args.step_stride), 1) != 0:
                    return callback_kwargs
                cur = {}
                for b in sorted(collector.maps.keys()):
                    if b % max(int(args.block_stride), 1) != 0:
                        continue
                    cur[b] = aggregate_head_map(collector.maps[b], head_mode=args.aggregate_head).clone()
                if cur:
                    step_cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = step_cache_dir / f"step_{step_i:04d}.pt"
                    torch.save(cur, cache_path)
                    step_block_maps[step_i] = cache_path
                return callback_kwargs

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
                    callback_on_step_end=on_step_end,
                ).images[0]
            collector.remove()

            out.save(out_img)
            if len(step_block_maps) == 0:
                cur = {}
                for b in sorted(collector.maps.keys()):
                    if b % max(int(args.block_stride), 1) != 0:
                        continue
                    cur[b] = aggregate_head_map(collector.maps[b], head_mode=args.aggregate_head).clone()
                if cur:
                    step_block_maps[0] = cur
            out_map = out_root / f"{key}_attn" / f"attention_step_block_grid.{args.save_format}"
            save_step_block_grid(
                step_block_maps=step_block_maps,
                out_path=out_map,
                ref_labels=ref_labels,
                image_dpi=int(args.image_dpi),
                save_format=args.save_format,
                head_mode=args.aggregate_head,
                step_stride=max(int(args.step_stride), 1),
                block_stride=max(int(args.block_stride), 1),
                panel_size=float(args.panel_size),
                attn_cmap=str(args.attn_cmap),
                attn_gamma=float(args.attn_gamma),
                high_attn_quantile=float(args.high_attn_quantile),
                high_attn_contour_color=str(args.high_attn_contour_color),
                high_attn_contour_width=float(args.high_attn_contour_width),
                region_alpha=float(args.region_alpha),
                boundary_linewidth=float(args.boundary_linewidth),
                q_tokens_full=int(collector.meta.get(sorted(collector.meta.keys())[0], {}).get("q_tokens_full", 0)) if collector.meta else 0,
                k_tokens_full=int(collector.meta.get(sorted(collector.meta.keys())[0], {}).get("k_tokens_full", 0)) if collector.meta else 0,
                q_sample_indices=[int(x) for x in collector.meta.get(sorted(collector.meta.keys())[0], {}).get("q_sample_indices", [])] if collector.meta else [],
                k_sample_indices=[int(x) for x in collector.meta.get(sorted(collector.meta.keys())[0], {}).get("k_sample_indices", [])] if collector.meta else [],
                has_encoder=bool(collector.meta.get(sorted(collector.meta.keys())[0], {}).get("has_encoder", False)) if collector.meta else False,
                text_tokens_est=int(text_tokens_est),
            )
            if args.save_attn_tensor:
                out_tensor = out_root / f"{key}_attn" / "attention_step_block_grid.pt"
                save_step_block_tensor(
                    step_block_maps=step_block_maps,
                    out_path=out_tensor,
                    key=key,
                    prompt=prompt,
                    ref_labels=ref_labels,
                    aggregate_head=str(args.aggregate_head),
                    step_stride=max(int(args.step_stride), 1),
                    block_stride=max(int(args.block_stride), 1),
                    q_tokens_full=int(collector.meta.get(sorted(collector.meta.keys())[0], {}).get("q_tokens_full", 0)) if collector.meta else 0,
                    k_tokens_full=int(collector.meta.get(sorted(collector.meta.keys())[0], {}).get("k_tokens_full", 0)) if collector.meta else 0,
                    q_sample_indices=[int(x) for x in collector.meta.get(sorted(collector.meta.keys())[0], {}).get("q_sample_indices", [])] if collector.meta else [],
                    k_sample_indices=[int(x) for x in collector.meta.get(sorted(collector.meta.keys())[0], {}).get("k_sample_indices", [])] if collector.meta else [],
                    has_encoder=bool(collector.meta.get(sorted(collector.meta.keys())[0], {}).get("has_encoder", False)) if collector.meta else False,
                    text_tokens_est=int(text_tokens_est),
                    collector_meta=collector.meta,
                )
            if collector.meta:
                sample_block = sorted(collector.meta.keys())[0]
                token_summary = out_root / f"{key}_attn" / "token_layout_summary.txt"
                save_token_layout_summary(
                    out_path=token_summary,
                    key=key,
                    prompt=prompt,
                    ref_labels=ref_labels,
                    collector=collector,
                    sample_block=sample_block,
                    step_block_maps=step_block_maps,
                    tokenizer=getattr(pipe, "tokenizer", None),
                )
            done += 1
            status = "done"
            print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} generated_image={out_img}")
            print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} fullmap={out_map}")
            if args.save_attn_tensor:
                print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} attn_tensor={out_root / f'{key}_attn' / 'attention_step_block_grid.pt'}")
            summary_path = out_root / f"{key}_attn" / "token_layout_summary.txt"
            if summary_path.exists():
                print(f"[SUMMARY][worker={worker_idx}][gpu={gpu}] key={key} token_layout={summary_path}")
            print(f"[SELF-CHECK][worker={worker_idx}][gpu={gpu}] key={key} {'PASS' if out_map.exists() else 'FAIL'}")
        except Exception as exc:
            failed += 1
            status = "failed"
            print(f"[FAIL][worker={worker_idx}][gpu={gpu}] key={key} {type(exc).__name__}: {exc}")
        finally:
            if collector is not None:
                try:
                    collector.remove()
                except Exception:
                    pass
            if step_cache_dir is not None and step_cache_dir.exists():
                shutil.rmtree(step_cache_dir, ignore_errors=True)
            if use_cuda:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            _emit_progress(progress_queue, status=status, key=key, worker_idx=worker_idx, gpu=gpu)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(done=done, skip=skipped, fail=failed)

    if pbar is not None:
        pbar.close()
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
            progress_queue=None,
            show_local_progress=True,
        )
    else:
        chunks = split_keys_round_robin(keys, len(gpu_list))
        tasks = []
        for worker_idx, (gpu, sub_keys) in enumerate(zip(gpu_list, chunks)):
            if not sub_keys:
                continue
            tasks.append((worker_idx, int(gpu), args, prompts, sub_keys, key_to_index))
        mp_ctx = mp.get_context("spawn")
        manager = mp_ctx.Manager()
        progress_queue = manager.Queue()
        progress_stats = {"done": 0, "skipped": 0, "failed": 0}
        with mp_ctx.Pool(processes=len(tasks)) as pool:
            async_results = [
                pool.apply_async(
                    run_inference_worker,
                    args=task,
                    kwds={"progress_queue": progress_queue, "show_local_progress": False},
                )
                for task in tasks
            ]
            with tqdm(total=len(keys), desc="total", dynamic_ncols=True) as pbar:
                processed = 0
                while processed < len(keys):
                    try:
                        event = progress_queue.get(timeout=1.0)
                        processed += 1
                        status = str(event.get("status", "failed"))
                        if status not in progress_stats:
                            progress_stats[status] = 0
                        progress_stats[status] += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            done=progress_stats.get("done", 0),
                            skip=progress_stats.get("skipped", 0),
                            fail=progress_stats.get("failed", 0),
                        )
                    except Empty:
                        if all(res.ready() for res in async_results):
                            break
                while processed < len(keys):
                    try:
                        event = progress_queue.get_nowait()
                    except Empty:
                        break
                    processed += 1
                    status = str(event.get("status", "failed"))
                    if status not in progress_stats:
                        progress_stats[status] = 0
                    progress_stats[status] += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        done=progress_stats.get("done", 0),
                        skip=progress_stats.get("skipped", 0),
                        fail=progress_stats.get("failed", 0),
                    )
            results = [res.get() for res in async_results]
        done = int(sum(x[0] for x in results))
        skipped = int(sum(x[1] for x in results))
        failed = int(sum(x[2] for x in results))

    print(f"[FINAL] done={done} skipped={skipped} failed={failed} total={len(keys)}")


if __name__ == "__main__":
    main()

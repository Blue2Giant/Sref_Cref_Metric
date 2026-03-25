#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Qwen-Image-Edit-2511 Attention FullMap 可视化")
    parser.add_argument("--prompts_json", required=True, help="id->prompt 的json文件")
    parser.add_argument("--cref_dir", required=True, help="内容参考图目录，文件名应为 {id}.png")
    parser.add_argument("--sref_dir", required=True, help="风格参考图目录，文件名应为 {id}.png")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--gpus", default="0", help='如 "0" 或 "0,1,2"，本脚本仅使用第一个GPU')
    parser.add_argument("--key_txt", required=True, help="txt文件，可包含多行key")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128, help="attention矩阵q/k最大采样token数")
    parser.add_argument("--aggregate-head", choices=["mean", "max"], default="mean")
    parser.add_argument("--aggregate-block", choices=["mean", "max"], default="mean")
    parser.add_argument("--step-stride", type=int, default=4, help="每隔多少步采样一个step")
    parser.add_argument("--block-stride", type=int, default=4, help="每隔多少个block采样一个")
    parser.add_argument("--panel-size", type=float, default=1.4, help="step-block拼图每个子图尺寸")
    parser.add_argument("--image-dpi", type=int, default=120)
    parser.add_argument("--save-format", choices=["png", "pdf"], default="png")
    parser.add_argument("--ref-labels", default="cref,sref", help="参考图名称，逗号分隔，按输入图顺序")
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


class AttentionCollector:
    def __init__(self, max_tokens: int):
        self.max_tokens = max(1, int(max_tokens))
        self.handles = []
        self.maps: Dict[int, torch.Tensor] = {}

    def _sample_tokens(self, n: int) -> torch.Tensor:
        if n <= self.max_tokens:
            return torch.arange(n, dtype=torch.long)
        return torch.linspace(0, n - 1, steps=self.max_tokens).long()

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
                q_idx = self._sample_tokens(attn.shape[1])
                k_idx = self._sample_tokens(attn.shape[2])
                attn = attn[:, q_idx][:, :, k_idx]
                self.maps[block_idx] = attn
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


def save_step_block_grid(
    step_block_maps: Dict[int, Dict[int, torch.Tensor]],
    out_path: Path,
    ref_labels: List[str],
    image_dpi: int,
    save_format: str,
    head_mode: str,
    step_stride: int,
    block_stride: int,
    panel_size: float,
):
    if len(step_block_maps) == 0:
        raise RuntimeError("没有可用的step-block attention")
    steps = sorted(step_block_maps.keys())
    block_ids = sorted({b for s in steps for b in step_block_maps[s].keys()})
    if len(block_ids) == 0:
        raise RuntimeError("没有可用的block attention")

    sample = step_block_maps[steps[0]][block_ids[0]]
    q_ranges = build_ref_ranges(int(sample.shape[0]), len(ref_labels))
    k_ranges = build_ref_ranges(int(sample.shape[1]), len(ref_labels))
    colors = plt.cm.get_cmap("tab20", max(1, len(ref_labels)))

    all_vals = []
    for s in steps:
        for b in block_ids:
            x = step_block_maps[s].get(b)
            if x is not None:
                all_vals.append(x)
    vmin = min(float(x.min().item()) for x in all_vals)
    vmax = max(float(x.max().item()) for x in all_vals)
    panel_size = max(0.8, float(panel_size))
    fig_w = max(10.0, len(block_ids) * panel_size)
    fig_h = max(8.0, len(steps) * panel_size)
    fig, axes = plt.subplots(len(steps), len(block_ids), figsize=(fig_w, fig_h), squeeze=False)
    im = None
    for ridx, step in enumerate(steps):
        for cidx, block_id in enumerate(block_ids):
            ax = axes[ridx][cidx]
            mat = step_block_maps[step].get(block_id)
            if mat is None:
                ax.axis("off")
                continue
            im = ax.imshow(mat.numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            for rr, qs, qe in q_ranges:
                if qe > qs:
                    ax.axhspan(qs, qe - 1, alpha=0.08, color=colors(rr))
            for rr, ks, ke in k_ranges:
                if ke > ks:
                    ax.axvspan(ks, ke - 1, alpha=0.08, color=colors(rr))
            ax.set_xticks([])
            ax.set_yticks([])
            if ridx == 0:
                ax.set_title(f"B{block_id}", fontsize=8)
            if cidx == 0:
                ax.set_ylabel(f"S{step}", fontsize=8)

    handles = []
    for ridx, (qs, qe), (ks, ke) in zip(range(len(ref_labels)), [(x[1], x[2]) for x in q_ranges], [(x[1], x[2]) for x in k_ranges]):
        handles.append(
            mpatches.Patch(
                color=colors(ridx),
                label=f"{ref_labels[ridx]} Q[{qs},{max(qs, qe-1)}] K[{ks},{max(ks, ke-1)}]",
            )
        )
    if handles:
        fig.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)), fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(
        f"Attention Grid (rows=step stride {step_stride}, cols=block stride {block_stride}, head={head_mode})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
):
    q_len = int(full_map.shape[0])
    k_len = int(full_map.shape[1])
    num_refs = len(ref_labels)
    q_ranges = build_ref_ranges(q_len, num_refs)
    k_ranges = build_ref_ranges(k_len, num_refs)
    colors = plt.cm.get_cmap("tab20", max(1, num_refs))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(full_map.numpy(), aspect="auto", cmap="viridis")
    for ridx, qs, qe in q_ranges:
        if qe > qs:
            ax.axhspan(qs, qe - 1, alpha=0.12, color=colors(ridx))
    for ridx, ks, ke in k_ranges:
        if ke > ks:
            ax.axvspan(ks, ke - 1, alpha=0.12, color=colors(ridx))
    xticks = []
    xlabels = []
    yticks = []
    ylabels = []
    for ridx, ks, ke in k_ranges:
        if ke > ks:
            xticks.append((ks + ke - 1) * 0.5)
            xlabels.append(f"K:{ref_labels[ridx]}")
    for ridx, qs, qe in q_ranges:
        if qe > qs:
            yticks.append((qs + qe - 1) * 0.5)
            ylabels.append(f"Q:{ref_labels[ridx]}")
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8)
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
    handles = []
    for ridx, (qs, qe), (ks, ke) in zip(range(len(ref_labels)), [(x[1], x[2]) for x in q_ranges], [(x[1], x[2]) for x in k_ranges]):
        handles.append(
            mpatches.Patch(
                color=colors(ridx),
                label=f"{ref_labels[ridx]} Q[{qs},{max(qs, qe-1)}] K[{ks},{max(ks, ke-1)}]",
            )
        )
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(f"Full Attention Map (head={head_mode}, block={block_mode})")
    ax.set_xlabel("key tokens")
    ax.set_ylabel("query tokens")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=image_dpi, bbox_inches="tight", format=save_format)
    plt.close(fig)


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
    keys = read_keys(args.key_txt)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    for idx, key in enumerate(keys):
        if key not in prompts:
            print(f"[SKIP] key不在prompts_json中，已跳过: {key}")
            skipped += 1
            continue
        cref_path = Path(args.cref_dir) / f"{key}.png"
        sref_path = Path(args.sref_dir) / f"{key}.png"
        if not cref_path.exists() or not sref_path.exists():
            print(f"[SKIP] 图片缺失: key={key} cref={cref_path.exists()} sref={sref_path.exists()}")
            skipped += 1
            continue

        prompt = prompts[key]
        cref = load_rgb(str(cref_path))
        sref = load_rgb(str(sref_path))
        images = [cref, sref]
        ref_labels = [x.strip() for x in str(args.ref_labels).split(",") if x.strip()]
        if len(ref_labels) < len(images):
            ref_labels += [f"ref{i}" for i in range(len(ref_labels), len(images))]
        ref_labels = ref_labels[: len(images)]

        collector = AttentionCollector(max_tokens=args.max_tokens)
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            collector.register(pipe.transformer)
        else:
            collector.register(pipe.unet if hasattr(pipe, "unet") else pipe)

        generator = torch.Generator(device=device).manual_seed(int(args.seed) + idx)
        step_block_maps: Dict[int, Dict[int, torch.Tensor]] = {}

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
                step_block_maps[step_i] = cur
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

        out_img = out_root / f"{key}.png"
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
        )
        done += 1
        print(f"[SUMMARY] key={key} generated_image={out_img}")
        print(f"[SUMMARY] key={key} fullmap={out_map}")
        print(f"[SELF-CHECK] key={key} {'PASS' if out_map.exists() else 'FAIL'}")

    print(f"[FINAL] done={done} skipped={skipped} total={len(keys)}")


if __name__ == "__main__":
    main()

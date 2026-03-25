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
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Qwen-Image-Edit-2511 Attention 可视化")
    parser.add_argument("--prompts_json", required=True, help="id->prompt 的json文件")
    parser.add_argument("--cref_dir", required=True, help="内容参考图目录，文件名应为 {id}.png")
    parser.add_argument("--sref_dir", required=True, help="风格参考图目录，文件名应为 {id}.png")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--gpus", default="0", help='如 "0" 或 "0,1,2"，本脚本仅使用第一个GPU')
    parser.add_argument("--key_txt", required=True, help="txt文件，第一行非空字符串作为唯一key")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256, help="attention矩阵q/k最大采样token数")
    parser.add_argument("--image-dpi", type=int, default=300)
    parser.add_argument("--save-format", choices=["png", "pdf"], default="png")
    parser.add_argument("--ref-labels", default="cref,sref", help="参考图名称，逗号分隔，按输入图顺序")
    parser.add_argument("--panel-size", type=float, default=1.35, help="拼图中单个block-head子图尺寸")
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


def build_ref_ranges(key_len: int, num_refs: int) -> List[Tuple[int, int, int]]:
    if num_refs <= 0:
        return []
    base = key_len // num_refs
    rem = key_len % num_refs
    out = []
    start = 0
    for i in range(num_refs):
        span = base + (1 if i < rem else 0)
        end = start + span
        out.append((i, start, end))
        start = end
    return out


def visualize_attention_maps(
    attn_maps: Dict[int, torch.Tensor],
    output_dir: Path,
    num_refs: int,
    ref_labels: List[str],
    save_format: str,
    image_dpi: int,
    panel_size: float,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(attn_maps) == 0:
        return 0
    colors = plt.cm.get_cmap("tab20", max(1, num_refs))
    blocks = sorted(attn_maps.keys())
    max_heads = max(int(attn_maps[b].shape[0]) for b in blocks)
    sample = attn_maps[blocks[0]]
    q_ranges = build_ref_ranges(int(sample.shape[1]), num_refs)
    k_ranges = build_ref_ranges(int(sample.shape[2]), num_refs)

    panel_size = max(0.6, float(panel_size))
    fig_w = max(10.0, max_heads * panel_size)
    fig_h = max(8.0, len(blocks) * panel_size)
    fig, axes = plt.subplots(len(blocks), max_heads, figsize=(fig_w, fig_h), squeeze=False)
    im = None
    for row_idx, block_idx in enumerate(blocks):
        block_map = attn_maps[block_idx]
        heads = int(block_map.shape[0])
        for col_idx in range(max_heads):
            ax = axes[row_idx][col_idx]
            if col_idx >= heads:
                ax.axis("off")
                continue
            mat = block_map[col_idx].numpy()
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            for ridx, qs, qe in q_ranges:
                if qe > qs:
                    ax.axhspan(qs, qe - 1, alpha=0.08, color=colors(ridx))
            for ridx, ks, ke in k_ranges:
                if ke > ks:
                    ax.axvspan(ks, ke - 1, alpha=0.08, color=colors(ridx))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"H{col_idx}", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"B{block_idx}", fontsize=8)

    legend_handles = []
    for ridx, (qs, qe), (ks, ke) in zip(range(len(ref_labels)), [(x[1], x[2]) for x in q_ranges], [(x[1], x[2]) for x in k_ranges]):
        legend_handles.append(
            mpatches.Patch(
                color=colors(ridx),
                label=f"{ref_labels[ridx]} Q[{qs},{max(qs, qe-1)}] K[{ks},{max(ks, ke-1)}]",
            )
        )
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)), fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
    fig.suptitle("Rows=Block, Cols=Head (attention score)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    name = f"attention_grid.{save_format}"
    fig.savefig(output_dir / name, dpi=image_dpi, bbox_inches="tight")
    plt.close(fig)
    return 1


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
    ref_labels = [x.strip() for x in str(args.ref_labels).split(",") if x.strip()]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    done = 0
    skipped = 0
    for idx, key in enumerate(keys):
        if key not in prompts:
            print(f"[SKIP] key不在prompts_json中，已跳过: {key}")
            skipped += 1
            continue
        prompt = prompts[key]

        cref_path = Path(args.cref_dir) / f"{key}.png"
        sref_path = Path(args.sref_dir) / f"{key}.png"
        if not cref_path.exists() or not sref_path.exists():
            print(f"[SKIP] 图片缺失: key={key} cref={cref_path.exists()} sref={sref_path.exists()}")
            skipped += 1
            continue

        cref = load_rgb(str(cref_path))
        sref = load_rgb(str(sref_path))
        images = [cref, sref]
        labels = ref_labels
        if len(labels) < len(images):
            labels = labels + [f"ref{i}" for i in range(len(labels), len(images))]
        labels = labels[: len(images)]

        collector = AttentionCollector(max_tokens=args.max_tokens)
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            collector.register(pipe.transformer)
        else:
            collector.register(pipe.unet if hasattr(pipe, "unet") else pipe)

        generator = torch.Generator(device=device).manual_seed(int(args.seed) + idx)
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
        collector.remove()

        out_path = out_root / f"{key}.png"
        out.save(out_path)
        output_dir = out_root / f"{key}_attn"
        saved_count = visualize_attention_maps(
            collector.maps,
            output_dir=output_dir,
            num_refs=len(images),
            ref_labels=labels,
            save_format=args.save_format,
            image_dpi=int(args.image_dpi),
            panel_size=float(args.panel_size),
        )

        files = list(output_dir.glob(f"*.{args.save_format}"))
        block_count = len(collector.maps)
        head_total = sum(int(x.shape[0]) for x in collector.maps.values())
        expected = 1
        ok = len(files) >= min(expected, saved_count)
        done += 1
        print(f"[SUMMARY] key={key} blocks={block_count} heads_total={head_total} refs={len(images)} labels={','.join(labels)}")
        print(f"[SUMMARY] key={key} saved_count={saved_count} files_on_disk={len(files)} expected={expected}")
        print(f"[SUMMARY] key={key} generated_image={out_path}")
        print(f"[SUMMARY] key={key} output_dir={output_dir}")
        print(f"[SELF-CHECK] key={key} {'PASS' if ok and len(files) > 0 else 'FAIL'}")
    print(f"[FINAL] done={done} skipped={skipped} total={len(keys)}")


if __name__ == "__main__":
    main()

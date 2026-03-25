#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Qwen-Image-Edit-2511 style增量低频抑制推理")
    parser.add_argument("--prompts_json", required=True, help="id->prompt 的json文件")
    parser.add_argument("--cref_dir", required=True, help="内容参考图目录，文件名应为 {id}.png")
    parser.add_argument("--sref_dir", required=True, help="风格参考图目录，文件名应为 {id}.png")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--gpus", default="0", help='如 "0" 或 "0,1,2"，本脚本仅使用第一个GPU')
    parser.add_argument("--key_txt", required=True, help="txt文件，可包含多行key")
    parser.add_argument("--negative-prompt", "--negative_prompt", dest="negative_prompt", default=" ")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--true-cfg-scale", "--true_cfg_scale", dest="true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment",
        choices=["raw", "suppress_lp", "suppress_hp", "lp_restore", "all"],
        default="lp_restore",
    )
    parser.add_argument("--lp-factor", type=int, default=4)
    parser.add_argument("--lp-mode", choices=["nearest", "bilinear", "bicubic"], default="bilinear")
    parser.add_argument("--alpha-hp", type=float, default=0.5)
    parser.add_argument("--beta-const", type=float, default=0.2)
    parser.add_argument("--beta-schedule", choices=["piecewise", "sigmoid"], default="piecewise")
    parser.add_argument("--early-ratio", type=float, default=0.35)
    parser.add_argument("--mid-ratio", type=float, default=0.7)
    parser.add_argument("--beta-early", type=float, default=0.0)
    parser.add_argument("--beta-mid", type=float, default=0.2)
    parser.add_argument("--beta-late", type=float, default=0.5)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--beta-r0", type=float, default=0.5)
    parser.add_argument("--beta-s", type=float, default=0.12)
    parser.add_argument("--metrics_jsonl", default="", help="可选，保存每步 rho 与 alpha/beta")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--disable-transformer-cache-context", action="store_true")
    parser.add_argument("--empty-cache-per-step", type=int, default=0)
    parser.add_argument("--offload-image-latents-to-cpu", action="store_true")
    parser.add_argument("--offload-prompt-embeds-to-cpu", action="store_true")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--enable-vae-slicing", action="store_true")
    parser.add_argument("--enable-vae-tiling", action="store_true")
    parser.add_argument("--attention-slicing", choices=["none", "auto", "max"], default="none")
    parser.add_argument("--device-map", choices=["none", "auto", "balanced", "sequential"], default="none")
    parser.add_argument("--max-memory-gpu", default="", help='逗号分隔，如 "70GiB,70GiB"')
    parser.add_argument("--max-memory-cpu", default="120GiB")
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


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def retrieve_timesteps(scheduler, num_inference_steps: int, device: torch.device, sigmas: List[float], mu: float):
    params = set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    kwargs = {"device": device}
    if "sigmas" in params:
        kwargs["sigmas"] = sigmas
    else:
        kwargs["num_inference_steps"] = num_inference_steps
    if "mu" in params:
        kwargs["mu"] = mu
    scheduler.set_timesteps(**kwargs)
    return scheduler.timesteps


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    width = math.sqrt(float(target_area) * ratio)
    height = width / ratio
    width = int(round(width / 32.0) * 32)
    height = int(round(height / 32.0) * 32)
    return width, height


def preprocess_plus_images(pipe, images: List[Image.Image]):
    condition_images = []
    vae_images = []
    vae_image_sizes = []
    for img in images:
        image_width, image_height = img.size
        ratio = float(image_width) / float(image_height)
        cond_w, cond_h = calculate_dimensions(384 * 384, ratio)
        vae_w, vae_h = calculate_dimensions(1024 * 1024, ratio)
        condition_images.append(pipe.image_processor.resize(img, cond_h, cond_w))
        vae_images.append(pipe.image_processor.preprocess(img, vae_h, vae_w).unsqueeze(2))
        vae_image_sizes.append((vae_w, vae_h))
    return condition_images, vae_images, vae_image_sizes


def lowpass_tokens(x: torch.Tensor, grid_h: int, grid_w: int, factor: int, mode: str):
    if factor <= 1:
        return x
    b, n, c = x.shape
    if n != grid_h * grid_w:
        return x
    y = x.transpose(1, 2).reshape(b, c, grid_h, grid_w)
    h2 = max(1, grid_h // factor)
    w2 = max(1, grid_w // factor)
    if mode == "nearest":
        y_lp = F.interpolate(F.interpolate(y, size=(h2, w2), mode=mode), size=(grid_h, grid_w), mode=mode)
    else:
        y_lp = F.interpolate(
            F.interpolate(y, size=(h2, w2), mode=mode, align_corners=False),
            size=(grid_h, grid_w),
            mode=mode,
            align_corners=False,
        )
    return y_lp.reshape(b, c, n).transpose(1, 2)


def beta_from_schedule(args, progress: float) -> float:
    if args.beta_schedule == "piecewise":
        if progress < float(args.early_ratio):
            return float(args.beta_early)
        if progress < float(args.mid_ratio):
            return float(args.beta_mid)
        return float(args.beta_late)
    z = (progress - float(args.beta_r0)) / max(float(args.beta_s), 1e-6)
    return float(args.beta_max) * (1.0 / (1.0 + math.exp(-z)))


def get_alpha_beta(args, progress: float, experiment: str) -> Tuple[float, float]:
    if experiment == "raw":
        return 1.0, 1.0
    if experiment == "suppress_lp":
        return 1.0, float(args.beta_const)
    if experiment == "suppress_hp":
        return float(args.alpha_hp), 1.0
    if experiment == "lp_restore":
        return 1.0, beta_from_schedule(args, progress)
    raise RuntimeError(f"不支持的 experiment: {experiment}")


def cfg_combine(pos_pred: torch.Tensor, neg_pred: torch.Tensor, true_cfg_scale: float):
    comb_pred = neg_pred + true_cfg_scale * (pos_pred - neg_pred)
    cond_norm = torch.norm(pos_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm.clamp_min(1e-6))


def compute_txt_seq_lens(mask: torch.Tensor):
    if mask is None:
        return None
    lens = mask.to(torch.int32).sum(dim=1).detach().cpu().tolist()
    return [int(x) for x in lens]


@torch.inference_mode()
def sample_with_style_lp_guidance(pipe, cref: Image.Image, sref: Image.Image, prompt: str, args, seed: int, experiment: str):
    device = pipe._execution_device
    batch_size = 1
    num_images_per_prompt = 1
    full_cond_images, full_vae_images, full_vae_sizes = preprocess_plus_images(pipe, [cref, sref])
    base_cond_images, base_vae_images, base_vae_sizes = preprocess_plus_images(pipe, [cref])
    width, height = full_vae_sizes[0]
    do_true_cfg = float(args.true_cfg_scale) > 1 and args.negative_prompt is not None

    prompt_embeds_full, prompt_mask_full = pipe.encode_prompt(
        image=full_cond_images,
        prompt=prompt,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=int(args.max_sequence_length),
    )
    prompt_embeds_base, prompt_mask_base = pipe.encode_prompt(
        image=base_cond_images,
        prompt=prompt,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=int(args.max_sequence_length),
    )

    neg_embeds_full = None
    neg_mask_full = None
    neg_embeds_base = None
    neg_mask_base = None
    if do_true_cfg:
        neg_embeds_full, neg_mask_full = pipe.encode_prompt(
            image=full_cond_images,
            prompt=args.negative_prompt,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=int(args.max_sequence_length),
        )
        neg_embeds_base, neg_mask_base = pipe.encode_prompt(
            image=base_cond_images,
            prompt=args.negative_prompt,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=int(args.max_sequence_length),
        )

    generator = torch.Generator(device=device).manual_seed(int(seed))
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, image_latents_full = pipe.prepare_latents(
        full_vae_images,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds_full.dtype,
        device,
        generator,
        latents=None,
    )
    latents, image_latents_base = pipe.prepare_latents(
        base_vae_images,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds_base.dtype,
        device,
        generator=None,
        latents=latents.clone(),
    )

    prompt_embeds_full_cpu = None
    prompt_mask_full_cpu = None
    prompt_embeds_base_cpu = None
    prompt_mask_base_cpu = None
    neg_embeds_full_cpu = None
    neg_mask_full_cpu = None
    neg_embeds_base_cpu = None
    neg_mask_base_cpu = None
    if args.offload_prompt_embeds_to_cpu:
        prompt_embeds_full_cpu = prompt_embeds_full.cpu()
        prompt_mask_full_cpu = prompt_mask_full.cpu()
        prompt_embeds_base_cpu = prompt_embeds_base.cpu()
        prompt_mask_base_cpu = prompt_mask_base.cpu()
        prompt_embeds_full = None
        prompt_mask_full = None
        prompt_embeds_base = None
        prompt_mask_base = None
        if do_true_cfg:
            neg_embeds_full_cpu = neg_embeds_full.cpu()
            neg_mask_full_cpu = neg_mask_full.cpu()
            neg_embeds_base_cpu = neg_embeds_base.cpu()
            neg_mask_base_cpu = neg_mask_base.cpu()
            neg_embeds_full = None
            neg_mask_full = None
            neg_embeds_base = None
            neg_mask_base = None

    image_latents_full_cpu = None
    image_latents_base_cpu = None
    if args.offload_image_latents_to_cpu:
        if image_latents_full is not None:
            image_latents_full_cpu = image_latents_full.cpu()
            image_latents_full = None
        if image_latents_base is not None:
            image_latents_base_cpu = image_latents_base.cpu()
            image_latents_base = None

    grid_h = height // pipe.vae_scale_factor // 2
    grid_w = width // pipe.vae_scale_factor // 2
    img_shapes_full = [
        [(1, grid_h, grid_w)] + [(1, h // pipe.vae_scale_factor // 2, w // pipe.vae_scale_factor // 2) for (w, h) in full_vae_sizes]
    ] * batch_size
    img_shapes_base = [
        [(1, grid_h, grid_w)] + [(1, h // pipe.vae_scale_factor // 2, w // pipe.vae_scale_factor // 2) for (w, h) in base_vae_sizes]
    ] * batch_size

    sigmas = np.linspace(1.0, 1 / int(args.steps), int(args.steps))
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps = retrieve_timesteps(pipe.scheduler, int(args.steps), device, sigmas=sigmas, mu=mu)
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(0)

    use_cache_context = hasattr(pipe.transformer, "cache_context") and (not args.disable_transformer_cache_context)
    rho_trace = []
    with pipe.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            progress = i / max(len(timesteps) - 1, 1)
            alpha, beta = get_alpha_beta(args, progress, experiment)
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
            image_latents_full_step = image_latents_full
            image_latents_base_step = image_latents_base
            if image_latents_full_step is None and image_latents_full_cpu is not None:
                image_latents_full_step = image_latents_full_cpu.to(device)
            if image_latents_base_step is None and image_latents_base_cpu is not None:
                image_latents_base_step = image_latents_base_cpu.to(device)

            prompt_embeds_full_step = prompt_embeds_full
            prompt_mask_full_step = prompt_mask_full
            prompt_embeds_base_step = prompt_embeds_base
            prompt_mask_base_step = prompt_mask_base
            neg_embeds_full_step = neg_embeds_full
            neg_mask_full_step = neg_mask_full
            neg_embeds_base_step = neg_embeds_base
            neg_mask_base_step = neg_mask_base
            if prompt_embeds_full_step is None and prompt_embeds_full_cpu is not None:
                prompt_embeds_full_step = prompt_embeds_full_cpu.to(device)
                prompt_mask_full_step = prompt_mask_full_cpu.to(device)
                prompt_embeds_base_step = prompt_embeds_base_cpu.to(device)
                prompt_mask_base_step = prompt_mask_base_cpu.to(device)
                if do_true_cfg:
                    neg_embeds_full_step = neg_embeds_full_cpu.to(device)
                    neg_mask_full_step = neg_mask_full_cpu.to(device)
                    neg_embeds_base_step = neg_embeds_base_cpu.to(device)
                    neg_mask_base_step = neg_mask_base_cpu.to(device)

            txt_seq_lens_full = compute_txt_seq_lens(prompt_mask_full_step)
            txt_seq_lens_base = compute_txt_seq_lens(prompt_mask_base_step)
            txt_seq_lens_full_neg = compute_txt_seq_lens(neg_mask_full_step) if do_true_cfg else None
            txt_seq_lens_base_neg = compute_txt_seq_lens(neg_mask_base_step) if do_true_cfg else None

            input_full = torch.cat([latents, image_latents_full_step], dim=1) if image_latents_full_step is not None else latents
            input_base = torch.cat([latents, image_latents_base_step], dim=1) if image_latents_base_step is not None else latents

            full_ctx = pipe.transformer.cache_context("style_full") if use_cache_context else nullcontext()
            with full_ctx:
                v_full = pipe.transformer(
                    hidden_states=input_full,
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_mask_full_step,
                    encoder_hidden_states=prompt_embeds_full_step,
                    txt_seq_lens=txt_seq_lens_full,
                    img_shapes=img_shapes_full,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            v_full = v_full[:, : latents.size(1)]

            base_ctx = pipe.transformer.cache_context("style_base") if use_cache_context else nullcontext()
            with base_ctx:
                v_base = pipe.transformer(
                    hidden_states=input_base,
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_mask_base_step,
                    encoder_hidden_states=prompt_embeds_base_step,
                    txt_seq_lens=txt_seq_lens_base,
                    img_shapes=img_shapes_base,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            v_base = v_base[:, : latents.size(1)]

            if do_true_cfg:
                neg_full_ctx = pipe.transformer.cache_context("style_full_neg") if use_cache_context else nullcontext()
                with neg_full_ctx:
                    v_full_neg = pipe.transformer(
                        hidden_states=input_full,
                        timestep=timestep,
                        guidance=None,
                        encoder_hidden_states_mask=neg_mask_full_step,
                        encoder_hidden_states=neg_embeds_full_step,
                        txt_seq_lens=txt_seq_lens_full_neg,
                        img_shapes=img_shapes_full,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                v_full_neg = v_full_neg[:, : latents.size(1)]
                v_full = cfg_combine(v_full, v_full_neg, float(args.true_cfg_scale))

                neg_base_ctx = pipe.transformer.cache_context("style_base_neg") if use_cache_context else nullcontext()
                with neg_base_ctx:
                    v_base_neg = pipe.transformer(
                        hidden_states=input_base,
                        timestep=timestep,
                        guidance=None,
                        encoder_hidden_states_mask=neg_mask_base_step,
                        encoder_hidden_states=neg_embeds_base_step,
                        txt_seq_lens=txt_seq_lens_base_neg,
                        img_shapes=img_shapes_base,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                v_base_neg = v_base_neg[:, : latents.size(1)]
                v_base = cfg_combine(v_base, v_base_neg, float(args.true_cfg_scale))

            dv_style = v_full - v_base
            dv_lp = lowpass_tokens(dv_style, grid_h, grid_w, int(args.lp_factor), args.lp_mode)
            dv_hp = dv_style - dv_lp
            v_tilde = v_base + alpha * dv_hp + beta * dv_lp

            dv_lp_n = torch.norm(dv_lp.reshape(dv_lp.shape[0], -1), dim=-1)
            dv_n = torch.norm(dv_style.reshape(dv_style.shape[0], -1), dim=-1).clamp_min(1e-6)
            rho = (dv_lp_n / dv_n).mean().item()
            rho_trace.append(
                {
                    "step": int(i),
                    "progress": float(progress),
                    "t": float(t.item()) if hasattr(t, "item") else float(t),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "rho_lp": float(rho),
                }
            )

            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(v_tilde, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)
            del input_full
            del input_base
            del v_full
            del v_base
            del dv_style
            del dv_lp
            del dv_hp
            del v_tilde
            if args.empty_cache_per_step > 0 and torch.cuda.is_available() and ((i + 1) % int(args.empty_cache_per_step) == 0):
                torch.cuda.empty_cache()
            progress_bar.update()

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std_inv = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / latents_std_inv + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    pipe.maybe_free_model_hooks()
    return image, rho_trace


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

    use_device_map = torch.cuda.is_available() and len(gpu_list) > 1 and str(args.device_map) != "none"
    max_memory = None
    if use_device_map and str(args.max_memory_gpu).strip():
        gpu_limits = [x.strip() for x in str(args.max_memory_gpu).split(",") if x.strip()]
        if len(gpu_limits) == 1:
            gpu_limits = gpu_limits * len(gpu_list)
        if len(gpu_limits) >= len(gpu_list):
            max_memory = {int(g): gpu_limits[i] for i, g in enumerate(gpu_list)}
            if str(args.max_memory_cpu).strip():
                max_memory["cpu"] = str(args.max_memory_cpu).strip()
    pipe_kwargs = {"torch_dtype": dtype}
    if use_device_map:
        pipe_kwargs["device_map"] = str(args.device_map)
        pipe_kwargs["low_cpu_mem_usage"] = True
        if max_memory is not None:
            pipe_kwargs["max_memory"] = max_memory
    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_name, **pipe_kwargs)
    if args.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if args.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if str(args.attention_slicing) == "auto" and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    elif str(args.attention_slicing) == "max" and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)
    if use_device_map and (args.enable_model_cpu_offload or args.enable_sequential_cpu_offload):
        print("[WARN] device_map模式下已忽略 enable_model_cpu_offload / enable_sequential_cpu_offload")
    if use_device_map:
        pass
    elif torch.cuda.is_available() and args.enable_sequential_cpu_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    elif torch.cuda.is_available() and args.enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    experiments = [args.experiment] if args.experiment != "all" else ["raw", "suppress_lp", "suppress_hp", "lp_restore"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl.strip() else None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_json)
    keys = read_keys(args.key_txt)
    done = 0
    skipped = 0
    for kidx, key in enumerate(keys):
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

        for eidx, exp in enumerate(experiments):
            image, rho_trace = sample_with_style_lp_guidance(
                pipe=pipe,
                cref=cref,
                sref=sref,
                prompt=prompt,
                args=args,
                seed=int(args.seed) + kidx * 1000 + eidx,
                experiment=exp,
            )
            name = f"{key}.png" if args.experiment != "all" else f"{key}.{exp}.png"
            out_path = out_dir / name
            image.save(out_path)
            rho_mean = sum(x["rho_lp"] for x in rho_trace) / max(len(rho_trace), 1)
            done += 1
            print(f"[DONE] key={key} exp={exp} saved={out_path} rho_mean={rho_mean:.6f}")

            if metrics_path is not None:
                with open(metrics_path, "a", encoding="utf-8") as f:
                    for row in rho_trace:
                        rec = {
                            "key": key,
                            "experiment": exp,
                            "step": row["step"],
                            "progress": row["progress"],
                            "t": row["t"],
                            "alpha": row["alpha"],
                            "beta": row["beta"],
                            "rho_lp": row["rho_lp"],
                            "lp_factor": int(args.lp_factor),
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[FINAL] done={done} skipped={skipped} total_keys={len(keys)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import multiprocessing as mp
import torch
from PIL import Image
from tqdm import tqdm


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


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Flux2KleinPipeline on Sref/Cref prompts.json (multi-GPU, resize cref to preferred ratio)"
    )
    p.add_argument(
        "--prompts_json",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark_new800/bench/sref/prompts.json",
        help="Path to prompts.json (id->prompt).",
    )
    p.add_argument(
        "--cref_dir",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark_new800/bench/sref/cref",
        help="Directory containing content reference images, e.g., 000015.png",
    )
    p.add_argument(
        "--sref_dir",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark_new800/bench/sref/sref",
        help="Directory containing style/reference images, e.g., 000015.png",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./flux2_klein_outputs_resize",
        help="Output directory to save generated images.",
    )

    # Model / inference
    p.add_argument(
        "--model_name",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/flux_klein-9B",
        help="Model id/path for Flux2KleinPipeline.",
    )
    p.add_argument("--steps", type=int, default=4, help="num_inference_steps (Klein distilled default is often 4)")
    p.add_argument("--guidance_scale", type=float, default=1.0, help="guidance_scale (Klein example uses 1.0)")

    p.add_argument(
        "--no_images",
        action="store_true",
        help="Disable passing images; run pure text-to-image with prompts only.",
    )
    p.add_argument(
        "--use_only_cref",
        action="store_true",
        help="If set, only pass cref as image (img2img). If not set, pass [cref, sref] as multi-reference.",
    )

    # Sampling / determinism
    p.add_argument("--seed", type=int, default=42, help="Base seed.")
    p.add_argument(
        "--seed_strategy",
        type=str,
        choices=["fixed", "per_id"],
        default="fixed",
        help='fixed: all samples use same seed; per_id: seed = base_seed + int(id).',
    )

    # Optional filtering
    p.add_argument("--limit", type=int, default=0, help="Run first N items only. 0 = all.")
    p.add_argument(
        "--ids",
        type=str,
        default="",
        help='Comma-separated ids to run, e.g. "000015,000010". Empty = all.',
    )

    # Resize control
    p.add_argument(
        "--no_resize_cref",
        action="store_true",
        help="Disable resizing cref to closest PREFERRED_KONTEXT_RESOLUTIONS aspect ratio.",
    )
    p.add_argument(
        "--input_resolution",
        type=str,
        default="",
        help='Override input resolution as "WIDTHxHEIGHT", e.g. "1024x1024".',
    )

    p.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable enable_model_cpu_offload(gpu_id=...). Saves VRAM but slower and uses more CPU RAM.",
    )

    p.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help='GPU ids to use, e.g. "0,1,2,3". If CUDA not available, runs single-process on CPU.',
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite outputs. Default: skip existing outputs.",
    )

    p.add_argument(
        "--save_jsonl",
        action="store_true",
        help="Save metadata jsonl. Multi-proc will write per-rank files: metadata.rank{r}.jsonl",
    )

    return p.parse_args()


def load_prompts(prompts_json: str) -> Dict[str, str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def safe_open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.copy()


def _lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def resize_cref(cref: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
    cref_w, cref_h = cref.size  # PIL: (W,H)
    aspect_ratio = cref_w / float(cref_h)

    _, target_w, target_h = min(
        (abs(aspect_ratio - (w / float(h))), w, h) for (w, h) in PREFERRED_KONTEXT_RESOLUTIONS
    )

    if (cref_w, cref_h) == (target_w, target_h):
        return cref, (target_w, target_h)

    resized = cref.resize((target_w, target_h), resample=_lanczos())
    return resized, (target_w, target_h)


def compute_seed(base_seed: int, seed_strategy: str, k: str) -> int:
    if seed_strategy == "per_id":
        try:
            return base_seed + int(k)
        except ValueError:
            return base_seed
    return base_seed


def choose_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def worker(rank: int, gpu_id: int, keys: List[str], prompts: Dict[str, str], args):
    # device
    if torch.cuda.is_available() and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")

    torch_dtype = choose_torch_dtype(device)

    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(args.model_name, torch_dtype=torch_dtype)


    if device.type == "cuda" and args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)

    meta_f = None
    if args.save_jsonl:
        meta_f = open(out_dir / f"metadata.rank{rank}.jsonl", "a", encoding="utf-8")

    missing = 0
    skipped = 0

    with torch.inference_mode():
        for k in tqdm(keys, desc=f"rank{rank} gpu{gpu_id}", position=rank, leave=True):
            out_path = out_dir / f"{k}.png"
            if (not args.overwrite) and out_path.exists():
                skipped += 1
                continue

            prompt = prompts[k]

            # decide sizes and images
            image_list: Optional[List[Image.Image]] = None
            content_size = (1024, 1024)

            if not args.no_images:
                cref_path = cref_dir / f"{k}.png"
                sref_path = sref_dir / f"{k}.png"
                if not cref_path.exists() or not sref_path.exists():
                    missing += 1
                    print(f"[WARN][rank{rank}] missing id={k}: cref={cref_path.exists()} sref={sref_path.exists()}")
                    continue

                cref = safe_open_rgb(cref_path)
                sref = safe_open_rgb(sref_path)

                if args.input_resolution:
                    try:
                        w_str, h_str = args.input_resolution.lower().split("x", 1)
                        override_size = (int(w_str), int(h_str))
                    except Exception:
                        raise ValueError(f"invalid --input_resolution {args.input_resolution}")
                    cref = cref.resize(override_size, resample=_lanczos())
                    content_size = override_size
                else:
                    # resize cref (按截图逻辑)
                    if not args.no_resize_cref:
                        cref, content_size = resize_cref(cref)  # (W,H)
                    else:
                        content_size = cref.size  # (W,H)

                if args.use_only_cref:
                    image_list = [cref]
                else:
                    # multi-reference：把 cref + sref 一起给 image
                    image_list = [cref, sref]
            else:
                content_size = (1024, 1024)
                image_list = None

            # seed
            sample_seed = compute_seed(args.seed, args.seed_strategy, k)
            gen = torch.Generator(device=device).manual_seed(sample_seed)

            out = pipe(
                prompt=prompt,
                image=image_list,
                height=content_size[1],
                width=content_size[0],
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=gen,
            ).images[0]

            if out.size != content_size:
                print(f"[WARN][rank{rank}] id={k} out.size={out.size} != target={content_size}")

            out.save(out_path)

            if meta_f is not None:
                record = {
                    "id": k,
                    "prompt": prompt,
                    "model_name": args.model_name,
                    "rank": rank,
                    "gpu_id": gpu_id,
                    "torch_dtype": str(torch_dtype),
                    "cpu_offload": bool(args.cpu_offload),
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "seed": sample_seed,
                    "no_images": bool(args.no_images),
                    "use_only_cref": bool(args.use_only_cref),
                    "content_size_wh": [content_size[0], content_size[1]],
                    "out_size_wh": [out.size[0], out.size[1]],
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_f.flush()

    if meta_f is not None:
        meta_f.close()

    print(
        f"[DONE][rank{rank} gpu{gpu_id}] processed={len(keys)} "
        f"missing={missing} skipped={skipped} out_dir={args.out_dir}"
    )


def main():
    args = parse_args()

    prompts = load_prompts(args.prompts_json)

    if args.ids.strip():
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        prompts = {k: prompts[k] for k in wanted if k in prompts}

    keys = sorted(prompts.keys())
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    if not torch.cuda.is_available():
        worker(rank=0, gpu_id=-1, keys=keys, prompts=prompts, args=args)
        return

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if len(gpus) == 0:
        raise ValueError("No GPUs parsed from --gpus")

    world = len(gpus)
    shards = [keys[r::world] for r in range(world)]

    ctx = mp.get_context("spawn")
    procs = []
    for r, gpu_id in enumerate(gpus):
        p = ctx.Process(target=worker, args=(r, gpu_id, shards[r], prompts, args))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"A worker process exited with code {p.exitcode}")


if __name__ == "__main__":
    main()

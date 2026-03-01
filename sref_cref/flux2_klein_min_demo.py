"""
python /data/benchmark_metrics/sref_cref/flux2_klein_min_demo.py \
  --content /data/benchmark_metrics/assets/content.webp \
  --style /data/benchmark_metrics/assets/style.webp \
  --prompt "Transfer the style while keeping the content." \
  --out /data/benchmark_metrics/sref_cref/flux2_klein_out.png \
  --model_id black-forest-labs/FLUX.2-klein-9B \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 4 \
  --guidance_scale 1.0 \
  --seed 0 \
  --cpu_offload
"""
import argparse
import inspect

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resolve_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def build_call_kwargs(pipe, prompt: str, content: Image.Image, style: Image.Image, args):
    params = inspect.signature(pipe.__call__).parameters
    kwargs = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator": args.generator,
    }
    if "image" in params:
        kwargs["image"] = [content, style]
    elif "images" in params:
        kwargs["images"] = [content, style]
    elif "image_prompt" in params:
        kwargs["image_prompt"] = [content, style]
    elif "ref_images" in params:
        kwargs["ref_images"] = [content, style]
    elif "ref_image" in params:
        kwargs["ref_image"] = [content, style]
    elif "image_1" in params and "image_2" in params:
        kwargs["image_1"] = content
        kwargs["image_2"] = style
    return kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-9B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--num_inference_steps", type=int, default=4)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu_offload", action="store_true")
    args = ap.parse_args()

    dtype = resolve_dtype(args.dtype)
    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    args.generator = torch.Generator(device=args.device).manual_seed(args.seed)
    content = load_image(args.content)
    style = load_image(args.style)
    call_kwargs = build_call_kwargs(pipe, args.prompt, content, style, args)
    image = pipe(**call_kwargs).images[0]
    image.save(args.out)


if __name__ == "__main__":
    main()

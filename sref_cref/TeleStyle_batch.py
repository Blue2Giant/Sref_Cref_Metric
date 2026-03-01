import argparse
import json
import os
from typing import Dict, List

import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from huggingface_hub import hf_hub_download


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: str) -> Dict[str, str]:
    items: Dict[str, str] = {}
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMG_EXTS:
            continue
        items[os.path.splitext(name)[0]] = path
    return items


class ImageStyleInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/mnt/jfs/model_zoo")
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source="huggingface",
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source="huggingface",
                    origin_file_pattern="text_encoder/model*.safetensors",
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source="huggingface",
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                ),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2509",
                download_source="huggingface",
                origin_file_pattern="processor/",
            ),
        )

        telestyle_base = os.getenv("TELESTYLE_DIR", "/mnt/jfs/model_zoo/Tele-AI/TeleStyle")
        telestyle_image = os.path.join(
            telestyle_base, "weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors"
        )
        speedup = os.path.join(
            telestyle_base, "weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
        )
        if not os.path.isfile(telestyle_image):
            telestyle_image = hf_hub_download(
                repo_id="Tele-AI/TeleStyle",
                filename="weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors",
                local_files_only=True,
            )
        if not os.path.isfile(speedup):
            speedup = hf_hub_download(
                repo_id="Tele-AI/TeleStyle",
                filename="weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                local_files_only=True,
            )

        self.pipe.load_lora(self.pipe.dit, telestyle_image)
        self.pipe.load_lora(self.pipe.dit, speedup)

    def inference(
        self,
        prompt: str,
        content_ref: str,
        style_ref: str,
        seed: int,
        num_inference_steps: int,
        minedge: int,
    ):
        w, h = Image.open(content_ref).convert("RGB").size
        minedge = minedge - minedge % 16
        if w > h:
            r = w / h
            h = minedge
            w = int(h * r) - int(h * r) % 16
        else:
            r = h / w
            w = minedge
            h = int(w * r) - int(w * r) % 16

        images = [
            Image.open(content_ref).convert("RGB").resize((w, h)),
            Image.open(style_ref).convert("RGB").resize((minedge, minedge)),
        ]

        image = self.pipe(
            prompt,
            edit_image=images,
            seed=seed,
            num_inference_steps=num_inference_steps,
            height=h,
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=1.0,
        )
        return image


def load_prompts(prompts_path: str) -> Dict[str, str]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("prompts.json must be a dict")
    return {str(k): str(v) for k, v in data.items()}


def main():
    ap = argparse.ArgumentParser("Batch TeleStyle inference")
    ap.add_argument("--cref_dir", required=True, help="Content reference images directory")
    ap.add_argument("--sref_dir", required=True, help="Style reference images directory")
    ap.add_argument("--prompts_json", required=True, help="prompts.json path")
    ap.add_argument("--output_dir", required=True, help="Output directory for generated images")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--minedge", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cref_dir = os.path.abspath(os.path.expanduser(args.cref_dir))
    sref_dir = os.path.abspath(os.path.expanduser(args.sref_dir))
    prompts_path = os.path.abspath(os.path.expanduser(args.prompts_json))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    if not os.path.isdir(cref_dir):
        raise SystemExit(f"Missing cref dir: {cref_dir}")
    if not os.path.isdir(sref_dir):
        raise SystemExit(f"Missing sref dir: {sref_dir}")
    if not os.path.isfile(prompts_path):
        raise SystemExit(f"Missing prompts.json: {prompts_path}")

    os.makedirs(output_dir, exist_ok=True)

    prompts = load_prompts(prompts_path)
    cref_map = list_images(cref_dir)
    sref_map = list_images(sref_dir)

    keys: List[str] = []
    for k in prompts.keys():
        if k in cref_map and k in sref_map:
            keys.append(k)
    if args.limit > 0:
        keys = keys[: args.limit]

    if not keys:
        raise SystemExit("No matching keys found between prompts/cref/sref")

    engine = ImageStyleInference()
    with torch.no_grad():
        for k in keys:
            prompt = prompts[k]
            out_path = os.path.join(output_dir, f"{k}.png")
            if os.path.isfile(out_path):
                continue
            image = engine.inference(
                prompt=prompt,
                content_ref=cref_map[k],
                style_ref=sref_map[k],
                seed=args.seed,
                num_inference_steps=args.steps,
                minedge=args.minedge,
            )
            image.save(out_path)


if __name__ == "__main__":
    main()

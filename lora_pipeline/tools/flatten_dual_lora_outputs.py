"""
展平双lora的结果
python /data/benchmark_metrics/lora_pipeline/tools/flatten_dual_lora_outputs.py \
    --input-root /mnt/jfs/loras_combine/illustrious_0322_dual_lora \
    --output-root /mnt/jfs/loras_combine/illustrious_dual_lora_see \
    --sample-model-count 200 \
    --image-subdir eval_images_with_negative_new \
    --convert-jpg \
    --jpg-quality 75

python /data/benchmark_metrics/lora_pipeline/tools/flatten_dual_lora_outputs.py \
    --input-root /mnt/jfs/loras_combine/flux_0321_dual_lora \
    --output-root /mnt/jfs/loras_combine/flux0321_dual_lora_see \
    --sample-model-count 200 \
    --image-subdir eval_images_with_negative_new \
    --convert-jpg \
    --jpg-quality 75
"""
import argparse
import io
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List
from PIL import Image


IMG_NAME_RE = re.compile(r"^(\d{5}_\d+)\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE)


def collect_model_dirs(input_root: Path, image_subdir: str) -> List[Path]:
    model_dirs: List[Path] = []
    for p in sorted(input_root.iterdir()):
        if not p.is_dir():
            continue
        img_dir = p / image_subdir
        if img_dir.is_dir():
            model_dirs.append(p)
    return model_dirs


def sample_models(model_dirs: List[Path], sample_count: int, seed: int) -> List[Path]:
    if sample_count <= 0 or sample_count >= len(model_dirs):
        return model_dirs
    rng = random.Random(seed)
    return sorted(rng.sample(model_dirs, sample_count), key=lambda x: x.name)


def flatten_outputs(
    input_root: Path,
    output_root: Path,
    image_subdir: str,
    sample_model_count: int,
    sample_seed: int,
    convert_jpg: bool,
    jpg_quality: int,
) -> Dict[str, object]:
    if not input_root.is_dir():
        raise RuntimeError(f"input_root 不存在或不是目录: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    all_model_dirs = collect_model_dirs(input_root, image_subdir)
    if not all_model_dirs:
        raise RuntimeError(f"在 {input_root} 下未找到包含 {image_subdir} 的 model 目录")

    selected_model_dirs = sample_models(all_model_dirs, sample_model_count, sample_seed)
    copied = 0
    skipped = 0
    per_slot_count: Dict[str, int] = {}
    selected_ids: List[str] = []

    for model_dir in selected_model_dirs:
        model_id = model_dir.name
        selected_ids.append(model_id)
        img_dir = model_dir / image_subdir
        for f in sorted(img_dir.iterdir()):
            if not f.is_file():
                continue
            m = IMG_NAME_RE.match(f.name)
            if not m:
                continue
            slot_name = m.group(1)
            ext = m.group(2).lower()
            slot_dir = output_root / slot_name
            slot_dir.mkdir(parents=True, exist_ok=True)
            try:
                if convert_jpg:
                    dst = slot_dir / f"{model_id}.jpg"
                    with f.open("rb") as src_f:
                        raw = src_f.read()
                    with Image.open(io.BytesIO(raw)) as img:
                        rgb = img.convert("RGB")
                        rgb.save(dst, format="JPEG", quality=int(jpg_quality), optimize=True)
                else:
                    dst = slot_dir / f"{model_id}.{ext}"
                    shutil.copy2(f, dst)
            except Exception as e:
                skipped += 1
                print(f"[WARN] skip invalid image: {f} err={e}")
                continue
            copied += 1
            per_slot_count[slot_name] = per_slot_count.get(slot_name, 0) + 1

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "image_subdir": image_subdir,
        "all_model_count": len(all_model_dirs),
        "selected_model_count": len(selected_model_dirs),
        "selected_model_ids": selected_ids,
        "copied_file_count": copied,
        "slot_count": len(per_slot_count),
        "sample_model_count": sample_model_count,
        "sample_seed": sample_seed,
        "convert_jpg": bool(convert_jpg),
        "jpg_quality": int(jpg_quality),
        "skipped_file_count": skipped,
    }
    (output_root / "flatten_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="将 dual_lora 输出按编号展平，并按 model_id 命名图片")
    parser.add_argument("--input-root", default="/mnt/jfs/loras_combine/illustrious_0322_dual_lora")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--image-subdir", default="eval_images_with_negative_new")
    parser.add_argument("--sample-model-count", type=int, default=0, help="随机采样 model 数量；0 表示全量")
    parser.add_argument("--sample-seed", type=int, default=42, help="随机采样种子")
    parser.add_argument("--convert-jpg", action="store_true", help="可选：输出统一转为 JPG")
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=85,
        help="JPG 质量(1-95，推荐 70-95；数值越高画质越好、体积越大)。仅在 --convert-jpg 时生效",
    )
    args = parser.parse_args()
    if not (1 <= int(args.jpg_quality) <= 95):
        raise RuntimeError("--jpg-quality 必须在 1~95")

    report = flatten_outputs(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        image_subdir=args.image_subdir,
        sample_model_count=args.sample_model_count,
        sample_seed=args.sample_seed,
        convert_jpg=bool(args.convert_jpg),
        jpg_quality=int(args.jpg_quality),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/LoraPipeline/utils/copy_one_lora_2see_flat.py \
  --eval-root s3://lanjinghong-data/loras_eval_illustrious_one_img_magic \
  --one-lora-root /mnt/jfs/loras_combine/illustrious_0318_one_lora \
  --out-root /mnt/jfs/loras_combine/illustrious_merged_eval_compare_illustrious_flat_v4a \
  --eval-subfolder eval_images_with_negative_new \
  --one-subfolder eval_images_with_negative_new \
  --only-model-ids /data/LoraPipeline/assets/illustrious_content_sample_final.txt \
  --limit-eval-per-model 4 \
  --limit-one-per-model 4

python /data/LoraPipeline/utils/copy_one_lora_2see_flat.py \
  --eval-root s3://lanjinghong-data/loras_eval_qwen \
  --one-lora-root /mnt/jfs/loras_combine/qwen_0316_one_lora \
  --out-root /mnt/jfs/loras_combine/qwen_merged_eval_compare_illustrious_flat_v4a \
  --eval-subfolder eval_images/ \
  --one-subfolder eval_images_with_negative_new \
  --only-model-ids /data/LoraPipeline/similarity_stats/qwen_ids.txt \
  --limit-eval-per-model 4 \
  --limit-one-per-model 4

python /data/LoraPipeline/utils/copy_one_lora_2see_flat.py \
  --eval-root s3://lanjinghong-data/loras_eval_flux_debug_1226 \
  --one-lora-root /mnt/jfs/loras_combine/flux_0321_one_lora \
  --out-root /mnt/jfs/loras_combine/flux_merged_eval_compare_flat_content_0321 \
  --eval-subfolder eval_images_with_negative/ \
  --one-subfolder eval_images_with_negative_new \
  --only-model-ids /data/LoraPipeline/assets/flux_content_sample.txt \
  --limit-eval-per-model 4 \
  --limit-one-per-model 4

python /data/LoraPipeline/utils/copy_one_lora_2see_flat.py \
  --eval-root s3://lanjinghong-data/loras_eval_flux_debug_1226 \
  --one-lora-root /mnt/jfs/loras_combine/flux_0318_one_lora \
  --out-root /mnt/jfs/loras_combine/flux_merged_eval_compare_flat_style_v4a \
  --eval-subfolder eval_images_with_negative/ \
  --one-subfolder eval_images_with_negative_new \
  --only-model-ids /data/LoraPipeline/assets/flux_style_1.txt \
  --limit-eval-per-model 4 \
  --limit-one-per-model 4

python /data/LoraPipeline/utils/copy_one_lora_2see_flat.py \
  --eval-root s3://lanjinghong-data/loras_eval_illustrious_one_img_magic \
  --one-lora-root /mnt/jfs/loras_combine/illustrious_0321_dual_lora \
  --out-root /mnt/jfs/loras_combine/illustrious_merged_eval_compare_illustrious_flat_v4b_new \
  --eval-subfolder eval_images_with_negative_new \
  --one-subfolder eval_images_with_negative_new \
  --only-model-ids /data/LoraPipeline/assets/illustrious_content_sample_final.txt \
  --limit-eval-per-model 4 \
  --limit-one-per-model 4 \
  --convert-jpg \
  --jpg-quality 82

python /data/LoraPipeline/utils/copy_one_lora_2see_flat.py \
  --eval-root s3://lanjinghong-data/loras_eval_qwen \
  --one-lora-root /mnt/jfs/loras_combine/qwen_0316_one_lora \
  --out-root /mnt/jfs/loras_combine/qwen_merged_eval_compare_illustrious_flat_v4b \
  --eval-subfolder eval_images/ \
  --one-subfolder eval_images_with_negative_new \
  --only-model-ids /data/LoraPipeline/similarity_stats/qwen_ids.txt \
  --limit-eval-per-model 4 \
  --limit-one-per-model 4

python /data/LoraPipeline/utils/copy_one_lora_2see_flat.py \
  --eval-root s3://lanjinghong-data/loras_eval_flux_debug_1226 \
  --one-lora-root /mnt/jfs/loras_combine/flux_0318_one_lora \
  --out-root /mnt/jfs/loras_combine/flux_merged_eval_compare_flat_content_v4b \
  --eval-subfolder eval_images_with_negative/ \
  --one-subfolder eval_images_with_negative_new \
  --only-model-ids /data/LoraPipeline/assets/flux_content_sample.txt \
  --limit-eval-per-model 4 \
  --limit-one-per-model 4
"""
import os
import re
import argparse
import io
from typing import List, Tuple, Optional
from PIL import Image

try:
    import megfile as mf
except Exception as e:
    raise RuntimeError(
        "需要 megfile 才能同时处理 s3:// 与本地路径。请先安装：pip install megfile\n"
        f"import megfile failed: {e}"
    )

IMAGE_EXTS_DEFAULT = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def smart_exists(path: str) -> bool:
    return mf.smart_exists(path)


def smart_isdir(path: str) -> bool:
    return mf.smart_isdir(path)


def smart_makedirs(path: str) -> None:
    mf.smart_makedirs(path, exist_ok=True)


def smart_listdir(path: str) -> List[str]:
    items = mf.smart_listdir(path)
    out: List[str] = []
    for x in items:
        x = str(x).rstrip("/")
        out.append(os.path.basename(x))
    return out


def smart_join(root: str, name: str) -> str:
    if root.endswith("/"):
        return root + name
    return root + "/" + name


def smart_copy(src: str, dst: str, overwrite: bool = True) -> None:
    if (not overwrite) and smart_exists(dst):
        return
    mf.smart_copy(src, dst)


def smart_copy_with_optional_jpg_compress(
    src: str,
    dst: str,
    overwrite: bool,
    convert_jpg: bool,
    jpg_quality: int,
) -> None:
    if (not overwrite) and smart_exists(dst):
        return
    if not convert_jpg:
        smart_copy(src, dst, overwrite=overwrite)
        return
    with mf.smart_open(src, "rb") as f:
        raw = f.read()
    with Image.open(io.BytesIO(raw)) as img:
        rgb = img.convert("RGB")
        out = io.BytesIO()
        rgb.save(out, format="JPEG", quality=int(jpg_quality), optimize=True)
    with mf.smart_open(dst, "wb") as f:
        f.write(out.getvalue())


def is_image(name: str, exts: List[str]) -> bool:
    low = name.lower()
    return any(low.endswith(x) for x in exts)


def parse_model_id(name: str) -> Optional[str]:
    m = re.match(r"^(\d+)$", (name or "").strip())
    return m.group(1) if m else None


def read_model_id_txt(path: str) -> List[str]:
    ids: List[str] = []
    if not path:
        return ids
    with mf.smart_open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"(\d+)", s)
            if m:
                ids.append(m.group(1))
            else:
                ids.append(s)
    return ids


def list_model_ids(root: str) -> List[str]:
    if (not smart_exists(root)) or (not smart_isdir(root)):
        raise RuntimeError(f"root 不存在或不是目录: {root}")
    names = smart_listdir(root)
    mids: List[str] = []
    for n in names:
        mid = parse_model_id(n)
        if mid is None:
            continue
        if smart_isdir(smart_join(root, n)):
            mids.append(mid)
    mids.sort()
    return mids


def list_images(dir_path: str, exts: List[str]) -> List[str]:
    if (not smart_exists(dir_path)) or (not smart_isdir(dir_path)):
        return []
    names = smart_listdir(dir_path)
    imgs = [n for n in names if is_image(n, exts)]
    imgs.sort()
    return imgs


def build_paths(
    eval_root: str,
    one_lora_root: str,
    model_id: str,
    eval_subfolder: str,
    one_subfolder: str,
) -> Tuple[str, str]:
    eval_dir = smart_join(smart_join(eval_root, model_id), eval_subfolder)
    one_dir = smart_join(smart_join(one_lora_root, model_id), one_subfolder)
    return eval_dir, one_dir


def main():
    ap = argparse.ArgumentParser(
        description="拷贝两路 model_id/eval_images_with_negative_new 图片到同一个新目录结构中（one_lora 侧无 one_lora 子目录）。"
    )
    ap.add_argument("--eval-root", required=True, help="A侧根目录，结构为 {eval_root}/{model_id}/{subfolder}/")
    ap.add_argument("--one-lora-root", required=True, help="B侧根目录，结构为 {one_lora_root}/{model_id}/{subfolder}/")
    ap.add_argument("--out-root", required=True, help="输出目录（本地或 s3://）")
    ap.add_argument("--exts", default=",".join(IMAGE_EXTS_DEFAULT), help="图片后缀，逗号分隔")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不实际拷贝")
    ap.add_argument("--only-model-ids", default=None, help="可选：逗号分隔 model_id，或 txt 路径")
    ap.add_argument("--limit-per-model", type=int, default=0, help="每个 model 最多处理多少对（<=0 不限制）")
    ap.add_argument("--limit-eval-per-model", type=int, default=0, help="每个 model 最多拷贝 eval 数（<=0 不限制）")
    ap.add_argument("--limit-one-per-model", type=int, default=0, help="每个 model 最多拷贝 one_lora 数（<=0 不限制）")
    ap.add_argument("--overwrite", action="store_true", help="同名覆盖")
    ap.add_argument("--convert-jpg", action="store_true", help="可选：输出时统一转为 JPG 压缩")
    ap.add_argument(
        "--jpg-quality",
        type=int,
        default=85,
        help="JPG 质量(1-95，推荐 70-95；数值越高体积越大、画质越好)。仅在 --convert-jpg 时生效",
    )
    ap.add_argument("--subfolder", default="eval_images_with_negative_new", help="兼容旧参数：同时作为 eval/one 两侧子目录")
    ap.add_argument("--eval-subfolder", default=None, help="可选：eval-root/{model_id} 下子目录名")
    ap.add_argument("--one-subfolder", default=None, help="可选：one-lora-root/{model_id} 下子目录名")

    args = ap.parse_args()
    eval_root = args.eval_root.rstrip("/")
    one_lora_root = args.one_lora_root.rstrip("/")
    out_root = args.out_root.rstrip("/")
    exts = [x.strip().lower() for x in args.exts.split(",") if x.strip()]
    dry_run = bool(args.dry_run)
    overwrite = True if not args.overwrite else True
    if not (1 <= int(args.jpg_quality) <= 95):
        raise RuntimeError("--jpg-quality 必须在 1~95")
    eval_subfolder = (args.eval_subfolder or args.subfolder).strip().strip("/")
    one_subfolder = (args.one_subfolder or args.subfolder).strip().strip("/")

    only_set = None
    if args.only_model_ids:
        if smart_exists(args.only_model_ids):
            ids = read_model_id_txt(args.only_model_ids)
            only_set = {x.strip() for x in ids if x.strip()}
            print(f"[INFO] 从 {args.only_model_ids} 读取到 {len(only_set)} 个 model_id")
        else:
            only_set = {x.strip() for x in args.only_model_ids.split(",") if x.strip()}

    mids_a = set(list_model_ids(eval_root))
    mids = sorted(mids_a)
    if only_set is not None:
        mids = [m for m in mids if m in only_set]
    if not mids:
        print("[WARN] 没有可处理的 model_id")
        return 0

    print(f"[INFO] 将处理 model_id 数量: {len(mids)}")
    smart_makedirs(out_root)

    total_pairs = 0
    total_eval_only = 0
    for model_id in mids:
        eval_dir, one_dir = build_paths(
            eval_root=eval_root,
            one_lora_root=one_lora_root,
            model_id=model_id,
            eval_subfolder=eval_subfolder,
            one_subfolder=one_subfolder,
        )
        eval_imgs = list_images(eval_dir, exts)
        one_imgs = list_images(one_dir, exts)

        if not eval_imgs:
            print(f"[WARN] {model_id}: eval 图片为空或目录不存在: {eval_dir}")
            continue
        if args.limit_eval_per_model > 0:
            eval_imgs = eval_imgs[: int(args.limit_eval_per_model)]
        if args.limit_one_per_model > 0:
            one_imgs = one_imgs[: int(args.limit_one_per_model)]

        if not one_imgs:
            print(f"[WARN] {model_id}: one_lora 图片为空或目录不存在: {one_dir}，仅拷贝 eval")
            n = len(eval_imgs)
            if args.limit_per_model > 0:
                n = min(n, int(args.limit_per_model))
            for i in range(n):
                ext = "jpg" if args.convert_jpg else "png"
                new_basename = f"{model_id}.{ext}"
                out_eval_dir = smart_join(out_root, f"eval_{i + 1}")
                smart_makedirs(out_eval_dir)
                src_eval = smart_join(eval_dir, eval_imgs[i])
                dst_eval = smart_join(out_eval_dir, new_basename)
                if dry_run:
                    print(f"[DRY] {src_eval} -> {dst_eval}")
                else:
                    smart_copy_with_optional_jpg_compress(
                        src=src_eval,
                        dst=dst_eval,
                        overwrite=overwrite,
                        convert_jpg=bool(args.convert_jpg),
                        jpg_quality=int(args.jpg_quality),
                    )
                total_eval_only += 1
            print(f"[OK] {model_id}: 仅拷贝 eval {n} 张")
            continue

        n = min(len(eval_imgs), len(one_imgs))
        if args.limit_per_model > 0:
            n = min(n, int(args.limit_per_model))
        if len(eval_imgs) != len(one_imgs):
            print(f"[WARN] {model_id}: 数量不一致 eval={len(eval_imgs)} vs one_lora={len(one_imgs)}；将配对前 {n} 张")

        for i in range(n):
            ext = "jpg" if args.convert_jpg else "png"
            new_basename = f"{model_id}.{ext}"
            out_eval_dir = smart_join(out_root, f"eval_{i + 1}")
            out_one_dir = smart_join(out_root, f"one_lora_{i + 1}")
            smart_makedirs(out_eval_dir)
            smart_makedirs(out_one_dir)
            src_eval = smart_join(eval_dir, eval_imgs[i])
            src_one = smart_join(one_dir, one_imgs[i])
            dst_eval = smart_join(out_eval_dir, new_basename)
            dst_one = smart_join(out_one_dir, new_basename)

            if dry_run:
                print(f"[DRY] {src_eval} -> {dst_eval}")
                print(f"[DRY] {src_one}  -> {dst_one}")
            else:
                smart_copy_with_optional_jpg_compress(
                    src=src_eval,
                    dst=dst_eval,
                    overwrite=overwrite,
                    convert_jpg=bool(args.convert_jpg),
                    jpg_quality=int(args.jpg_quality),
                )
                smart_copy_with_optional_jpg_compress(
                    src=src_one,
                    dst=dst_one,
                    overwrite=overwrite,
                    convert_jpg=bool(args.convert_jpg),
                    jpg_quality=int(args.jpg_quality),
                )
            total_pairs += 1

        print(f"[OK] {model_id}: 拷贝配对完成 {n} 对")

    print(f"[DONE] 总共拷贝配对图片对数: {total_pairs}")
    print(f"[DONE] 仅拷贝 eval（未拷贝 one_lora）数量: {total_eval_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

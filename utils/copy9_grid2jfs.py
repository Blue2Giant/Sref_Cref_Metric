#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把一个远程 S3 桶/前缀下的所有图片递归拷贝到本地 JFS，并统一保存为 .jpg

依赖：
  pip install pillow tqdm megfile

用法示例：
  python3 s3_images_to_jfs_jpg.py \
    --src "s3://your-bucket/some/prefix" \
    --dst "/mnt/jfs/your_dir/images_jpg" \
    --workers 32 \
    --quality 92 \
    --keep-structure \
    --overwrite

python /data/LoraPipeline/utils/copy9_grid2jfs.py --src s3://lanjinghong-data/loras_eval_illustrious_9grids_new --dst "/mnt/jfs/illustrious_9grid" --workers 32  --new-model-txt "/data/LoraPipeline/output/illustrious_new_ids.txt"

python /data/LoraPipeline/utils/copy9_grid2jfs.py --src s3://lanjinghong-data/loras_eval_sdxl_9grids_new --dst "/mnt/jfs/9grid/sxdxl_9grid" --workers 32  --new-model-txt "/data/LoraPipeline/output/sdxl_new_ids.txt"

python /data/LoraPipeline/utils/copy9_grid2jfs.py --src /mnt/jfs/9grid/illustrious_9grid --dst /mnt/jfs/9grid/sxdxl_9grid_filtered --workers 32  --new-model-txt "/data/LoraPipeline/output/sdxl_new_ids.txt"


说明：
- 默认递归扫描 src 下所有文件，过滤常见图片后缀
- 输出文件名：保持相对路径结构（--keep-structure），并把后缀统一改成 .jpg
- PNG/WebP 等带透明通道的图会用白底铺平再转 JPG
- 若源文件本身是 jpeg 且未要求 resize，可选择不重编码直接 copy（--passthrough-jpeg）
"""

import os
import re
import sys
import uuid
import argparse
from io import BytesIO
from typing import Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from PIL import Image

from megfile.smart import (
    smart_listdir,
    smart_exists,
    smart_open as mopen,
    smart_copy as mcopy,
)

Image.MAX_IMAGE_PIXELS = None  # 防止大图报错


IMG_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_remote_path(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("oss://")


def join_path(root: str, *parts: str) -> str:
    root = root.rstrip("/")
    out = root
    for p in parts:
        out = out + "/" + str(p).lstrip("/")
    return out


def norm_dir(path: str) -> str:
    return path.rstrip("/") + "/"


def safe_makedirs_local(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def guess_is_dir_entry(entry: str) -> bool:
    # megfile 的 listdir 在一些实现里会用 "xxx/" 表示“目录占位”
    return entry.endswith("/")


def get_basename(p: str) -> str:
    return os.path.basename(p.rstrip("/"))


def get_ext(p: str) -> str:
    return os.path.splitext(get_basename(p))[1].lower()


def iter_remote_files_recursive(prefix: str) -> Iterable[str]:
    """
    递归遍历 s3://bucket/prefix 下的所有对象（尽量兼容 listdir 返回相对名或绝对名）
    """
    prefix = prefix.rstrip("/")
    stack = [prefix]

    while stack:
        cur = stack.pop()
        try:
            items = smart_listdir(cur)
        except Exception as e:
            print(f"[WARN] listdir failed: {cur} err={e}")
            continue

        for it in items:
            it = str(it)
            # it 可能是相对名，也可能已经是 s3://... 全路径
            if is_remote_path(it):
                full = it.rstrip("/")
                name = get_basename(it)
            else:
                name = it
                full = join_path(cur, it).rstrip("/")

            # 目录：优先用 it/ 或 full/ 规则判断
            if guess_is_dir_entry(name) or guess_is_dir_entry(full):
                stack.append(full.rstrip("/"))
                continue

            # S3 的“目录”不一定有占位对象，这里再做一次 exists 判断（可选）
            # 如果某些实现里 listdir 返回了“子前缀名”，但 exists(full+"/") 才算目录，
            # 这里我们用一个启发式：如果没有扩展名、且 smart_exists(full+"/") 为真，就当目录
            ext = get_ext(name)
            if not ext:
                try:
                    if smart_exists(full + "/"):
                        stack.append(full.rstrip("/"))
                        continue
                except Exception:
                    pass

            yield full


def compute_rel_path(src_full: str, src_root: str) -> str:
    """
    计算 src_full 相对于 src_root 的相对路径（用于 --keep-structure）
    """
    src_root = norm_dir(src_root)
    if src_full.startswith(src_root):
        rel = src_full[len(src_root):]
        return rel.lstrip("/")
    # 兜底：只用 basename
    return get_basename(src_full)


def flatten_alpha_to_rgb(img: Image.Image) -> Image.Image:
    """
    把带透明通道的图铺到白底，转成 RGB
    """
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg.convert("RGB")

    if img.mode == "P" and ("transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg.convert("RGB")

    return img.convert("RGB")


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS)


def save_jpg_atomic(img: Image.Image, out_path: str, quality: int) -> None:
    tmp_path = out_path + f".tmp_{uuid.uuid4().hex}.jpg"
    try:
        img.save(tmp_path, format="JPEG", quality=int(quality), optimize=True)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def should_accept(ext: str, exts: set, include_re: Optional[re.Pattern], exclude_re: Optional[re.Pattern], full: str) -> bool:
    if ext.lower() not in exts:
        return False
    if include_re and not include_re.search(full):
        return False
    if exclude_re and exclude_re.search(full):
        return False
    return True


def process_one(
    src_full: str,
    src_root: str,
    dst_root: str,
    keep_structure: bool,
    overwrite: bool,
    quality: int,
    max_side: int,
    passthrough_jpeg: bool,
    reencode_jpeg: bool,
) -> Tuple[str, str]:
    """
    返回 (status, message)
      status: ok | skip | fail
    """
    try:
        if keep_structure:
            rel = compute_rel_path(src_full, src_root)
        else:
            rel = get_basename(src_full)

        base_noext = os.path.splitext(rel)[0]
        rel_out = base_noext + ".jpg"

        out_path = os.path.join(dst_root, rel_out)
        out_dir = os.path.dirname(out_path)
        safe_makedirs_local(out_dir)

        if (not overwrite) and os.path.exists(out_path):
            return "skip", out_path

        src_ext = get_ext(src_full)

        # 如果源本来就是 jpeg 且不 resize，允许直接 copy（省 CPU）
        if passthrough_jpeg and (not reencode_jpeg) and (max_side <= 0) and src_ext in (".jpg", ".jpeg"):
            # mcopy 支持 remote -> local
            mcopy(src_full, out_path, overwrite=True)
            return "ok", out_path

        # 读远程 -> PIL -> 转 JPG
        with mopen(src_full, "rb") as f:
            data = f.read()

        img = Image.open(BytesIO(data))
        img.load()  # 把数据读进来，避免 BytesIO 关闭后出错

        img = flatten_alpha_to_rgb(img)
        img = resize_max_side(img, max_side=max_side)

        save_jpg_atomic(img, out_path, quality=quality)
        return "ok", out_path

    except Exception as e:
        return "fail", f"{src_full} -> {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help='S3 源路径，例如 "s3://bucket/prefix"')
    ap.add_argument("--dst", required=True, help='JFS 本地目录，例如 "/mnt/jfs/xxx"')
    ap.add_argument("--workers", type=int, default=16, help="并发线程数（主要是 IO）")
    ap.add_argument("--quality", type=int, default=92, help="JPG 质量 1-100")
    ap.add_argument("--max-side", type=int, default=0, help="最大边缩放（0=不缩放）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    ap.add_argument("--keep-structure", action="store_true", help="保持 src 的相对目录结构")
    ap.add_argument("--include", default=None, help="只处理匹配该正则的路径（可选）")
    ap.add_argument("--exclude", default=None, help="跳过匹配该正则的路径（可选）")
    ap.add_argument("--exts", default=",".join(sorted(IMG_EXTS_DEFAULT)), help="允许的图片后缀，逗号分隔")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少张（0=不限制）")

    ap.add_argument("--passthrough-jpeg", action="store_true", help="源为 jpeg 且不缩放时直接 copy（不重编码）")
    ap.add_argument("--reencode-jpeg", action="store_true", help="即使源为 jpeg 也强制重编码")
    ap.add_argument("--new-model-txt", default=None, help="输出本次新增的 model_id 到该 txt 文件")

    args = ap.parse_args()

    if args.new_model_txt:
        # 清空或创建该文件，确保只包含“这次”新增的
        try:
            with open(args.new_model_txt, "w", encoding="utf-8") as f:
                pass
        except Exception as e:
            print(f"[WARN] cannot create new_model_txt: {e}")

    src_root = args.src.rstrip("/")
    dst_root = os.path.abspath(args.dst)
    safe_makedirs_local(dst_root)

    exts = {("." + x.lower().lstrip(".")) for x in args.exts.split(",") if x.strip()}
    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    # 列出全部文件（为了进度条有 total，这里会先收集到内存；如果特别大你再说我给你改成流式）
    print(f"[INFO] scanning: {src_root}")
    all_files: List[str] = []
    for f in iter_remote_files_recursive(src_root):
        ext = get_ext(f)
        if should_accept(ext, exts, include_re, exclude_re, f):
            all_files.append(f)
            if args.limit and len(all_files) >= args.limit:
                break

    print(f"[INFO] candidate images: {len(all_files)}")
    if not all_files:
        print("[WARN] no images found.")
        return

    ok = 0
    skip = 0
    fail = 0

    failed_log = os.path.join(dst_root, "failed.txt")
    copied_log = os.path.join(dst_root, "copied.txt")

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = []
        for src_full in all_files:
            futures.append(
                ex.submit(
                    process_one,
                    src_full,
                    src_root,
                    dst_root,
                    bool(args.keep_structure),
                    bool(args.overwrite),
                    int(args.quality),
                    int(args.max_side),
                    bool(args.passthrough_jpeg),
                    bool(args.reencode_jpeg),
                )
            )

        for fu in tqdm(as_completed(futures), total=len(futures), unit="img"):
            status, msg = fu.result()
            if status == "ok":
                ok += 1
                try:
                    with open(copied_log, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                    
                    if args.new_model_txt:
                        # msg is out_path, get filename stem
                        model_id = os.path.splitext(os.path.basename(msg))[0]
                        with open(args.new_model_txt, "a", encoding="utf-8") as f:
                            f.write(model_id + "\n")
                except Exception:
                    pass
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                try:
                    with open(failed_log, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass

    print(f"[DONE] ok={ok}, skip={skip}, fail={fail}")
    print(f"[LOG] copied: {copied_log}")
    print(f"[LOG] failed: {failed_log}")


if __name__ == "__main__":
    main()

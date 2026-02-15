#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_9grid_from_s3.py

功能：
1) 遍历 src_root 下的每个 model_id
2) 进入 <model_id>/<subdir>（默认 eval_images_with_negative）收集图片
3) 随机挑选 9 张“可成功解码”的图片（损坏的丢弃）
4) 拼成 3x3 九宫格（不缩放，只做 padding 对齐），保存到 out_root
5) 若凑不齐 9 张：将 model_id 及信息写入 missing_txt
6) 若提供 filter_txt：只处理 filter_txt 中列出的 model_id

输出：
- 九宫格图片：<out_root>/<model_id>.png （默认 PNG）
- 可选：bad_img_txt 记录损坏图片路径
- missing_txt 记录凑不齐的 model_id

示例：
python /data/LoraPipeline/utils/direct_copy.py  --src-root s3://lanjinghong-data/loras_eval_flux  --out-root s3://lanjinghong-data/loras_eval_flux_9grids_new_0009 --missing-txt /data/LoraPipeline/output/missing.txt   --filter-model-ids-txt /data/LoraPipeline/output/check_failed_125.txt
python /data/LoraPipeline/utils/direct_copy.py  --src-root s3://lanjinghong-data/loras_eval_flux_debug_1226  --out-root s3://lanjinghong-data/loras_eval_flux_9grids_new_1226_debug --missing-txt /data/LoraPipeline/output/missing.txt   --filter-model-ids-txt /data/LoraPipeline/output/check_failed_125.txt
python /data/LoraPipeline/utils/direct_copy.py  --src-root s3://lanjinghong-data/loras_eval_sdxl_one_img_magic  --out-root s3://lanjinghong-data/loras_eval_sdxl_9grids_new --missing-txt /data/LoraPipeline/output/missing_sdxl_new.txt --workers 8  --required-resolution 1024x1024
python /data/LoraPipeline/utils/direct_copy.py  --src-root s3://lanjinghong-data/loras_eval_illustrious_one_img_magic --out-root s3://lanjinghong-data/loras_eval_illustrious_9grids_new --missing-txt /data/LoraPipeline/output/missing_illustrious_new.txt --workers 8   --required-resolution 1024x1024

只处理指定 model_id 列表：
python make_9grid_from_s3.py \
  --src-root s3://lanjinghong-data/loras_eval_flux \
  --out-root /data/9grids \
  --missing-txt /data/missing.txt \
  --filter-model-ids-txt /data/model_ids.txt
"""

import os
import io
import sys
import re
import json
import time
import random
import argparse
import posixpath
import zlib
from typing import List, Optional, Tuple, Dict

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# =========================
# 统一路径 join（兼容 s3:// 与本地）
# =========================
def path_join(base: str, *parts: str) -> str:
    if base.startswith("s3://"):
        base2 = base.rstrip("/")
        tail = "/".join(p.strip("/") for p in parts if p is not None and p != "")
        return f"{base2}/{tail}" if tail else base2
    return os.path.join(base, *parts)

def path_dirname(p: str) -> str:
    if p.startswith("s3://"):
        # s3://bucket/a/b -> s3://bucket/a
        m = re.match(r"^(s3://[^/]+)(/.*)?$", p)
        if not m:
            return p
        prefix = m.group(1)
        rest = (m.group(2) or "").rstrip("/")
        if not rest:
            return prefix
        rest_dir = posixpath.dirname(rest)
        if rest_dir == "/":
            return prefix
        return prefix + rest_dir
    return os.path.dirname(p)

def is_image_path(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in IMG_EXTS

# =========================
# 后端：优先 megfile，否则 boto3
# =========================
class StorageBackend:
    def listdir(self, path: str) -> List[str]:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def write_bytes(self, path: str, data: bytes) -> None:
        raise NotImplementedError

    def makedirs(self, path: str) -> None:
        raise NotImplementedError

def get_backend() -> StorageBackend:
    # 1) megfile 优先
    try:
        from megfile.smart import (
            smart_listdir,
            smart_exists,
            smart_open as smart_open,
            smart_makedirs,
        )
        class MegfileBackend(StorageBackend):
            def listdir(self, path: str) -> List[str]:
                return list(smart_listdir(path))

            def exists(self, path: str) -> bool:
                return bool(smart_exists(path))

            def read_bytes(self, path: str) -> bytes:
                with smart_open(path, "rb") as f:
                    return f.read()

            def write_bytes(self, path: str, data: bytes) -> None:
                parent = path_dirname(path)
                self.makedirs(parent)
                with smart_open(path, "wb") as f:
                    f.write(data)

            def makedirs(self, path: str) -> None:
                try:
                    smart_makedirs(path, exist_ok=True)
                except TypeError:
                    try:
                        smart_makedirs(path)
                    except Exception as e:
                        if "File exists" in str(e):
                            return
                        raise
                except Exception as e:
                    if "File exists" in str(e):
                        return
                    raise

        return MegfileBackend()
    except Exception:
        pass

    # 2) boto3 fallback（仅对 s3:// 有意义，本地依旧用文件系统）
    try:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3")

        def parse_s3(s3_path: str) -> Tuple[str, str]:
            # s3://bucket/key
            assert s3_path.startswith("s3://")
            x = s3_path[len("s3://"):]
            bucket, _, key = x.partition("/")
            return bucket, key

        class Boto3Backend(StorageBackend):
            def listdir(self, path: str) -> List[str]:
                if not path.startswith("s3://"):
                    # local
                    try:
                        return os.listdir(path)
                    except FileNotFoundError:
                        return []
                bucket, prefix = parse_s3(path.rstrip("/") + "/")
                # 用 Delimiter='/' 模拟“目录”
                paginator = s3.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
                out = []
                for page in pages:
                    for cp in page.get("CommonPrefixes", []):
                        # 返回“子目录名”
                        sub = cp["Prefix"][len(prefix):].rstrip("/")
                        if sub:
                            out.append(sub)
                    for obj in page.get("Contents", []):
                        name = obj["Key"][len(prefix):]
                        if name and "/" not in name:
                            out.append(name)
                return out

            def exists(self, path: str) -> bool:
                if not path.startswith("s3://"):
                    return os.path.exists(path)
                bucket, key = parse_s3(path)
                try:
                    s3.head_object(Bucket=bucket, Key=key)
                    return True
                except ClientError:
                    # 也可能是 prefix（目录），用 list_objects 探测
                    bucket, prefix = parse_s3(path.rstrip("/") + "/")
                    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
                    return "Contents" in resp

            def read_bytes(self, path: str) -> bytes:
                if not path.startswith("s3://"):
                    with open(path, "rb") as f:
                        return f.read()
                bucket, key = parse_s3(path)
                obj = s3.get_object(Bucket=bucket, Key=key)
                return obj["Body"].read()

            def write_bytes(self, path: str, data: bytes) -> None:
                if not path.startswith("s3://"):
                    parent = os.path.dirname(path)
                    os.makedirs(parent, exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(data)
                    return
                bucket, key = parse_s3(path)
                # boto3 不需要 makedirs
                s3.put_object(Bucket=bucket, Key=key, Body=data)

            def makedirs(self, path: str) -> None:
                if not path.startswith("s3://"):
                    os.makedirs(path, exist_ok=True)
                # s3 无需创建目录

        return Boto3Backend()
    except Exception as e:
        raise RuntimeError(
            "既没找到 megfile，也没找到 boto3。请安装其一：pip install megfile 或 pip install boto3"
        ) from e

# =========================
# 图片读取与校验（损坏直接丢弃）
# =========================
def load_image_safely(backend: StorageBackend, img_path: str) -> Optional[Image.Image]:
    try:
        b = backend.read_bytes(img_path)
        bio = io.BytesIO(b)

        # 先 verify（快速校验结构），再 reopen（verify 后图像对象不可用）
        im = Image.open(bio)
        im.verify()

        bio2 = io.BytesIO(b)
        im2 = Image.open(bio2)
        im2.load()  # 强制解码，进一步排雷
        # 统一转换到 RGB，避免 PNG RGBA/P 模式 paste 的坑（不算缩放）
        if im2.mode not in ("RGB", "RGBA"):
            im2 = im2.convert("RGB")
        return im2
    except Exception:
        return None

# =========================
# 组九宫格（不缩放，只 padding 对齐）
# =========================
def make_3x3_grid(images: List[Image.Image], gap: int = 0, bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    assert len(images) == 9
    # 3x3
    grid = [[images[r * 3 + c] for c in range(3)] for r in range(3)]
    col_w = [0, 0, 0]
    row_h = [0, 0, 0]
    for r in range(3):
        for c in range(3):
            w, h = grid[r][c].size
            col_w[c] = max(col_w[c], w)
            row_h[r] = max(row_h[r], h)

    W = sum(col_w) + gap * 2  # 两条竖缝
    H = sum(row_h) + gap * 2  # 两条横缝
    canvas = Image.new("RGB", (W, H), color=bg)

    y = 0
    for r in range(3):
        x = 0
        for c in range(3):
            im = grid[r][c]
            w, h = im.size
            # 居中贴到该 cell 中
            ox = x + (col_w[c] - w) // 2
            oy = y + (row_h[r] - h) // 2
            if im.mode == "RGBA":
                canvas.paste(im, (ox, oy), mask=im.split()[-1])
            else:
                canvas.paste(im, (ox, oy))
            x += col_w[c] + (gap if c < 2 else 0)
        y += row_h[r] + (gap if r < 2 else 0)

    return canvas

# =========================
# 列出某个目录下所有图片（可选递归）
# =========================
def list_images(backend: StorageBackend, folder: str, recursive: bool = False) -> List[str]:
    results: List[str] = []
    stack = [folder.rstrip("/")]

    while stack:
        cur = stack.pop()
        try:
            names = backend.listdir(cur)
        except Exception:
            continue

        for name in names:
            p = path_join(cur, name)
            if is_image_path(p):
                results.append(p)
            elif recursive:
                # 尝试把它当目录继续深入（失败就忽略）
                try:
                    _ = backend.listdir(p)
                    stack.append(p)
                except Exception:
                    pass

        if not recursive:
            break

    return results

# =========================
# 列出 src_root 下 model_id（默认只看一级）
# =========================
def list_model_ids(backend: StorageBackend, src_root: str) -> List[str]:
    try:
        items = backend.listdir(src_root.rstrip("/"))
        # 一般就是 model_id 前缀
        # 过滤掉明显不是 id 的隐藏文件等
        items = [x.strip("/") for x in items if x and not x.startswith(".")]
        return items
    except Exception:
        return []

def read_model_id_filter(filter_txt: str) -> List[str]:
    ids = []
    with open(filter_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids

def write_lines(path: str, lines: List[str]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

# =========================
# 多进程 Worker
# =========================
class WorkerResult:
    def __init__(self, mid, status, log_msg, missing_info=None, bad_imgs=None):
        self.mid = mid
        self.status = status  # "OK", "MISS"
        self.log_msg = log_msg
        self.missing_info = missing_info
        self.bad_imgs = bad_imgs or []

def process_one_model(mid: str, args) -> WorkerResult:
    # 每个进程独立初始化 backend
    backend = get_backend()

    # 检查输出是否存在
    out_path = path_join(args.out_root, f"{mid}.{args.fmt}")
    if not args.overwrite and backend.exists(out_path):
        return WorkerResult(mid, "SKIP", f"exists {out_path}")

    if args.seed is not None:
        # 简单做个确定性 seed
        random.seed(args.seed + zlib.crc32(mid.encode("utf-8")))
    
    folder = path_join(args.src_root, mid, args.subdir)
    
    # 检查目录
    if not backend.exists(folder) and not backend.exists(folder.rstrip("/") + "/"):
        return WorkerResult(mid, "MISS", f"missing_folder {folder}", f"{mid}\tmissing_folder\t{folder}")

    paths = list_images(backend, folder, recursive=args.recursive)
    if not paths:
        return WorkerResult(mid, "MISS", f"no_images {folder}", f"{mid}\tno_images\t{folder}")

    random.shuffle(paths)

    chosen_imgs: List[Image.Image] = []
    bad_logs = []

    for p in paths:
        if len(chosen_imgs) >= 9:
            break
        im = load_image_safely(backend, p)
        if im is None:
            if args.bad_img_txt is not None:
                bad_logs.append(f"{mid}\t{p}")
            continue
        
        # 检查分辨率
        if args.req_res_tuple is not None:
            if im.size != args.req_res_tuple:
                # 分辨率不符，跳过
                continue

        chosen_imgs.append(im)

    if len(chosen_imgs) < 9:
        return WorkerResult(
            mid, 
            "MISS", 
            f"valid={len(chosen_imgs)} need=9 folder={folder}", 
            f"{mid}\tvalid={len(chosen_imgs)}\tneed=9\tfolder={folder}",
            bad_logs
        )

    grid = make_3x3_grid(chosen_imgs, gap=args.gap)

    # 保存
    out_path = path_join(args.out_root, f"{mid}.{args.fmt}")
    bio = io.BytesIO()
    save_fmt = "JPEG" if args.fmt in ("jpg", "jpeg") else "PNG"
    if save_fmt == "JPEG":
        grid.save(bio, format=save_fmt, quality=95)
    else:
        grid.save(bio, format=save_fmt)
    
    backend.write_bytes(out_path, bio.getvalue())
    
    return WorkerResult(mid, "OK", f"-> {out_path}", None, bad_logs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, help="例如 s3://lanjinghong-data/loras_eval_flux")
    ap.add_argument("--subdir", default="eval_images_with_negative_new", help="默认 eval_images_with_negative")
    ap.add_argument("--out-root", required=True, help="九宫格输出目录（可 s3:// 或本地）")
    ap.add_argument("--missing-txt", required=True, help="凑不齐9张的 model_id 记录到此 txt（本地路径）")
    ap.add_argument("--bad-img-txt", default=None, help="可选：记录损坏图片路径到此 txt（本地路径）")
    ap.add_argument("--filter-model-ids-txt", default=None, help="可选：仅处理该 txt 每行的 model_id")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    ap.add_argument("--recursive", action="store_true", help="递归搜索 subdir 下所有图片")
    ap.add_argument("--gap", type=int, default=0, help="九宫格 cell 之间的间隙像素（默认0）")
    ap.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg"], help="输出格式")
    ap.add_argument("--max-models", type=int, default=0, help="只处理前 N 个（0 表示不限制）")
    ap.add_argument("--workers", type=int, default=8, help="多进程并发数（默认8）")
    ap.add_argument("--overwrite", action="store_true", help="是否覆盖已存在的输出文件")
    ap.add_argument("--required-resolution", default=None, help="可选：限制图片分辨率，格式 WxH (例如 1024x1024)。不符合的图片将被忽略。")
    args = ap.parse_args()

    # 解析分辨率参数
    args.req_res_tuple = None
    if args.required_resolution:
        try:
            w_str, h_str = args.required_resolution.lower().split("x")
            args.req_res_tuple = (int(w_str), int(h_str))
            print(f"[INFO] 仅使用分辨率为 {args.req_res_tuple} 的图片")
        except Exception:
            print(f"[ERR] 分辨率参数格式错误: {args.required_resolution}，应为 WxH (如 1024x1024)")
            sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    backend = get_backend()

    # 读取过滤列表
    if args.filter_model_ids_txt:
        wanted = set(read_model_id_filter(args.filter_model_ids_txt))
    else:
        wanted = None

    model_ids = list_model_ids(backend, args.src_root)
    model_ids = sorted(model_ids)

    if wanted is not None:
        model_ids = [m for m in model_ids if m in wanted]

    if args.max_models and args.max_models > 0:
        model_ids = model_ids[:args.max_models]

    missing_logs: List[str] = []
    bad_logs: List[str] = []

    total = len(model_ids)
    print(f"[INFO] models to process: {total}, workers: {args.workers}")

    # 多进程处理
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_model, mid, args): mid for mid in model_ids}
        
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            mid = futures[fut]
            try:
                res = fut.result()
                if res.status == "OK":
                    print(f"[OK] {done_count}/{total} {res.mid} {res.log_msg}")
                elif res.status == "SKIP":
                    print(f"[SKIP] {done_count}/{total} {res.mid} {res.log_msg}")
                else:
                    print(f"[MISS] {done_count}/{total} {res.mid} {res.log_msg}")
                
                if res.missing_info:
                    missing_logs.append(res.missing_info)
                if res.bad_imgs:
                    bad_logs.extend(res.bad_imgs)
            except Exception as e:
                print(f"[ERR] {done_count}/{total} {mid} exception: {e}")
                missing_logs.append(f"{mid}\texception\t{str(e)}")

    # 写 txt（追加模式）
    if missing_logs:
        write_lines(args.missing_txt, missing_logs)
        print(f"[INFO] missing written: {len(missing_logs)} -> {args.missing_txt}")
    else:
        print("[INFO] no missing models.")

    if args.bad_img_txt is not None and bad_logs:
        write_lines(args.bad_img_txt, bad_logs)
        print(f"[INFO] bad images written: {len(bad_logs)} -> {args.bad_img_txt}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import re
import argparse
from typing import Iterable, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
"""
python /data/LoraPipeline/utils/copy_pair_images_to_local.py \
  --src-root s3://lanjinghong-data/loras_combine/flux_0111 \
  --dst-root /mnt/jfs/loras_combine/flux_0125 \
  --workers 32
"""
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from megfile.smart import (
        smart_listdir,
        smart_exists,
        smart_copy,
        smart_makedirs,
        smart_isdir,
    )
except Exception:
    smart_listdir = None
    smart_exists = None
    smart_copy = None
    smart_makedirs = None
    smart_isdir = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_remote_path(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("oss://")


def join_path(root: str, name: str) -> str:
    return root.rstrip("/") + "/" + name.lstrip("/")


def ensure_parent_dir_local(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def ensure_parent_dir_smart(path: str):
    if smart_makedirs is None:
        return
    parent = os.path.dirname(path.rstrip("/")) + "/"
    try:
        smart_makedirs(parent, exist_ok=True)
    except TypeError:
        try:
            smart_makedirs(parent)
        except Exception as e:
            if "File exists" in str(e):
                return
            raise
    except Exception as e:
        if "File exists" in str(e):
            return
        raise


def smart_copy_compat(src: str, dst: str, overwrite: bool):
    if (not overwrite) and smart_exists is not None and smart_exists(dst):
        return
    if is_remote_path(src) or is_remote_path(dst):
        if smart_copy is None:
            raise RuntimeError("megfile.smart not available for s3/oss copy")
        ensure_parent_dir_smart(dst)
        smart_copy(src, dst)
        return
    if (not overwrite) and os.path.exists(dst):
        return
    ensure_parent_dir_local(dst)
    import shutil
    shutil.copy2(src, dst)


def strip_prefix(path: str, prefix: str) -> str:
    prefix = prefix.rstrip("/")
    if path.startswith(prefix + "/"):
        return path[len(prefix) + 1 :]
    if path == prefix:
        return ""
    return path


def read_pair_list(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    pairs: Set[str] = set()
    try:
        if is_remote_path(path):
            if smart_exists is None:
                raise RuntimeError("megfile.smart not available")
            if not smart_exists(path):
                return pairs
            from megfile.smart import smart_open as mopen
            with mopen(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            if not os.path.exists(path):
                return pairs
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
    except Exception:
        return pairs

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        nums = re.findall(r"\d+", s)
        if len(nums) >= 2:
            pairs.add(f"{nums[0]}__{nums[1]}")
        elif "__" in s:
            pairs.add(s)
    return pairs


def iter_all_files(root: str) -> Iterable[str]:
    root = root.rstrip("/")
    if is_remote_path(root):
        if smart_listdir is None:
            raise RuntimeError("megfile.smart not available for s3/oss list")
        stack = [root]
        while stack:
            cur = stack.pop()
            try:
                names = smart_listdir(cur)
            except Exception:
                if smart_exists and smart_exists(cur):
                    yield cur
                continue
            for name in names:
                p = join_path(cur, name)
                is_dir = False
                if smart_isdir is not None:
                    try:
                        is_dir = bool(smart_isdir(p))
                    except Exception:
                        is_dir = False
                else:
                    is_dir = str(name).endswith("/")
                if is_dir:
                    stack.append(p.rstrip("/"))
                else:
                    yield p
        return
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def list_pair_dirs(root: str) -> List[str]:
    root = root.rstrip("/")
    dirs: List[str] = []
    if is_remote_path(root):
        if smart_listdir is None:
            raise RuntimeError("megfile.smart not available for s3/oss list")
        try:
            names = smart_listdir(root)
        except Exception:
            return []
        for name in names:
            p = join_path(root, name)
            is_dir = False
            if smart_isdir is not None:
                try:
                    is_dir = bool(smart_isdir(p))
                except Exception:
                    is_dir = False
            else:
                is_dir = str(name).endswith("/")
            if is_dir:
                dirs.append(p.rstrip("/"))
    else:
        if not os.path.isdir(root):
            return []
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                dirs.append(p)
    dirs.sort()
    return dirs


def list_files_in_dir(root: str) -> List[str]:
    files: List[str] = []
    if is_remote_path(root):
        if smart_listdir is None:
            raise RuntimeError("megfile.smart not available for s3/oss list")
        try:
            names = smart_listdir(root)
        except Exception:
            return []
        for name in names:
            p = join_path(root, name)
            is_dir = False
            if smart_isdir is not None:
                try:
                    is_dir = bool(smart_isdir(p))
                except Exception:
                    is_dir = False
            else:
                is_dir = str(name).endswith("/")
            if not is_dir:
                files.append(p)
    else:
        if not os.path.isdir(root):
            return []
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                files.append(p)
    return files


def copy_one_image(
    img_path: str,
    src_root: str,
    dst_root: str,
    overwrite: bool,
):
    rel = strip_prefix(img_path, src_root).lstrip("/").replace("\\", "/")
    dst_path = join_path(dst_root, rel)
    smart_copy_compat(img_path, dst_path, overwrite)

    base, _ = os.path.splitext(img_path)
    json_src = base + ".json"
    if smart_exists is not None:
        exists_json = smart_exists(json_src) if is_remote_path(json_src) else os.path.exists(json_src)
    else:
        exists_json = os.path.exists(json_src)
    if exists_json:
        json_rel = strip_prefix(json_src, src_root).lstrip("/").replace("\\", "/")
        json_dst = join_path(dst_root, json_rel)
        smart_copy_compat(json_src, json_dst, overwrite)
        return dst_path, json_dst
    return dst_path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, help="远程桶路径")
    ap.add_argument("--dst-root", required=True, help="本地输出根目录")
    ap.add_argument("--pair-ids", default=None, help="可选：pair 列表 txt")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    src_root = args.src_root.rstrip("/")
    dst_root = args.dst_root.rstrip("/")
    pairs = read_pair_list(args.pair_ids)

    if is_remote_path(src_root) and smart_exists is not None:
        if not smart_exists(src_root):
            raise SystemExit(f"src-root 不存在: {src_root}")
    elif not os.path.exists(src_root):
        raise SystemExit(f"src-root 不存在: {src_root}")

    os.makedirs(dst_root, exist_ok=True)

    candidates: List[str] = []
    if args.recursive:
        it_scan = iter_all_files(src_root)
        if tqdm is not None:
            it_scan = tqdm(it_scan, desc="Scanning", unit="file")
        for p in it_scan:
            ext = os.path.splitext(p)[1].lower()
            if ext not in IMG_EXTS:
                continue
            rel = strip_prefix(p, src_root).lstrip("/").replace("\\", "/")
            pair_dir = rel.split("/", 1)[0] if rel else ""
            if pairs and pair_dir not in pairs:
                continue
            candidates.append(p)
    else:
        pair_dirs = list_pair_dirs(src_root)
        if pairs:
            pair_dirs = [p for p in pair_dirs if os.path.basename(p.rstrip("/")) in pairs]
        it_dirs = pair_dirs
        if tqdm is not None:
            it_dirs = tqdm(pair_dirs, desc="Scanning pairs", unit="dir")
        for d in it_dirs:
            for p in list_files_in_dir(d):
                ext = os.path.splitext(p)[1].lower()
                if ext in IMG_EXTS:
                    candidates.append(p)

    print(f"[INFO] images to copy: {len(candidates)}")
    if not candidates:
        return

    def worker(img_path: str):
        return copy_one_image(img_path, src_root, dst_root, args.overwrite)

    if args.workers <= 1:
        it_copy = candidates
        if tqdm is not None:
            it_copy = tqdm(candidates, desc="Copying", unit="file")
        for p in it_copy:
            img_dst, json_dst = worker(p)
            print(f"[COPY] {img_dst}")
            if json_dst:
                print(f"[COPY] {json_dst}")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, p) for p in candidates]
        it = as_completed(futures)
        if tqdm is not None:
            it = tqdm(it, total=len(futures), unit="file", desc="Copying")
        for fut in it:
            img_dst, json_dst = fut.result()
            print(f"[COPY] {img_dst}")
            if json_dst:
                print(f"[COPY] {json_dst}")


if __name__ == "__main__":
    main()

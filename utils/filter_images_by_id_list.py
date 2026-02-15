#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_images_by_id_list.py

读取一个 txt（每行一个 model_id），在 src（本地或 s3://）下递归扫描图片，
只复制“文件名匹配 model_id”的图片到 dst，并保留原有嵌套结构。

用法示例：
python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/flux_content_final.txt --src /mnt/jfs/9grid/flux_9grid/   --dst /data/LoraPipeline/output/flux_0111_triplets_subset_for_human_content \

python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/flux_style_final.txt --src /mnt/jfs/9grid/flux_9grid/   --dst /data/LoraPipeline/output/flux_0111_triplets_subset_for_human_style \

python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/flux_style_1.txt  --src /mnt/jfs/9grid/flux_9grid/  --dst /data/LoraPipeline/output/flux_0111_triplets_subset_for_human_style \

python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/output/sdxl_new_ids.txt  --src /mnt/jfs/9grid/sxdxl_9grid/  --dst /data/LoraPipeline/output/sdxl_0111_triplets_subset \

python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/flux_content_human.txt  --src /mnt/jfs/9grid/flux_9grid/  --dst /mnt/jfs/9grid/flux_9grid/flux_content_human_subset_20250116


python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/output/illustrious_new_ids.txt --src /mnt/jfs/9grid/illustrious_9grid/  --dst  /mnt/jfs/9grid/illustrious_9grid_new/illustrious_0111_triplets_subset

python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/sdxl_content_2.txt --src /mnt/jfs/9grid/sxdxl_9grid  --dst  /mnt/jfs/9grid/sdxl_9grid_content
python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/illustrious_style_1.txt --src /mnt/jfs/9grid/illustrious_9grid  --dst  /mnt/jfs/9grid/illustrious_9grid_style_1
python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/illustrious_content_1.txt --src /mnt/jfs/9grid/illustrious_9grid  --dst  /mnt/jfs/9grid/illustrious_9grid_content_1

python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/illustrious_style_1.txt --src /mnt/jfs/9grid/illustrious_9grid  --dst  /mnt/jfs/9grid/illustrious_9grid_style
python /data/LoraPipeline/utils/filter_images_by_id_list.py --ids_txt /data/LoraPipeline/assets/illustrious_content_1.txt --src /mnt/jfs/9grid/illustrious_9grid  --dst  /mnt/jfs/9grid/illustrious_9grid_content

  --workers 64 \
  --only_images

如果你的 txt 在 s3 上：
--ids_txt s3://bucket/path/model_ids.txt

匹配规则：
--match stem      # 默认：去掉扩展名的文件名 == model_id
--match filename  # 完整文件名（含扩展名）== model_id
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional, Set, Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ---------- optional deps ----------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ---------- megfile ----------
try:
    from megfile.smart import (
        smart_listdir,
        smart_exists,
        smart_makedirs,
        smart_copy,
        smart_open as mopen,
    )
    try:
        from megfile.smart import smart_isdir  # 某些版本有
    except Exception:
        smart_isdir = None  # type: ignore
except Exception:
    smart_listdir = None  # type: ignore
    smart_exists = None  # type: ignore
    smart_makedirs = None  # type: ignore
    smart_copy = None  # type: ignore
    smart_isdir = None  # type: ignore
    mopen = None  # type: ignore


def is_s3_path(p: str) -> bool:
    return p.startswith("s3://")


def norm_root(p: str) -> str:
    return p.rstrip("/")


def join_path(root: str, name: str) -> str:
    # smart_listdir 可能返回相对名，也可能返回带 / 的“目录前缀”
    if name.startswith("s3://") or os.path.isabs(name):
        return name.rstrip("/")
    if root.endswith("/"):
        return (root + name).rstrip("/")
    return (root + "/" + name).rstrip("/")


def strip_prefix(path: str, prefix: str) -> str:
    path = path.replace("\\", "/")
    prefix = prefix.replace("\\", "/")
    prefix_n = norm_root(prefix) + "/"
    if path == norm_root(prefix):
        return ""
    if not path.startswith(prefix_n):
        raise ValueError(f"path not under prefix: path={path} prefix={prefix}")
    return path[len(prefix_n):]


def stem_and_ext(p: str) -> Tuple[str, str]:
    base = os.path.basename(p)
    stem, ext = os.path.splitext(base)
    return stem, ext.lower()


def is_image_path(p: str) -> bool:
    _, ext = stem_and_ext(p)
    return ext in IMG_EXTS


def exists_any(p: str) -> bool:
    if is_s3_path(p):
        if smart_exists is None:
            raise RuntimeError("megfile.smart.smart_exists 不可用，无法判断 s3:// 是否存在")
        return bool(smart_exists(p))
    return os.path.exists(p)


def ensure_parent_dir_local(dst_path: str) -> None:
    parent = os.path.dirname(dst_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_parent_dir_smart(dst_path: str) -> None:
    parent = os.path.dirname(dst_path).replace("\\", "/")
    if not parent:
        return
    if smart_makedirs is None:
        raise RuntimeError("megfile.smart.smart_makedirs 不可用")
    smart_makedirs(parent)


def read_bytes_any(path: str) -> bytes:
    if is_s3_path(path):
        if mopen is None:
            raise RuntimeError("megfile.smart.smart_open (mopen) 不可用，无法读取 s3://")
        with mopen(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def load_ids_from_txt(txt_path: str) -> Set[str]:
    raw = read_bytes_any(txt_path)
    text = raw.decode("utf-8-sig")  # 兼容 BOM
    ids: Set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # 允许用户 txt 里写了 xxx.png 这种：这里不强行去扩展名，
        # 因为你可能用 --match filename；如果你用 --match stem，
        # 下面的匹配函数会用 stem 对比。
        ids.add(s)
    return ids


def iter_all_files(root: str) -> Iterable[str]:
    """递归遍历 root 下所有文件路径（本地或 s3://）"""
    root = norm_root(root)

    if is_s3_path(root):
        if smart_listdir is None:
            raise RuntimeError("需要 megfile 才能遍历 s3:// 目录（smart_listdir 不可用）")

        stack = [root]
        while stack:
            cur = stack.pop()
            try:
                names = smart_listdir(cur)
            except Exception:
                if exists_any(cur):
                    yield cur
                continue

            for name in names:
                p = join_path(cur, name)

                # 判断目录：优先 smart_isdir，否则用 name 是否以 / 结尾
                is_dir = False
                if smart_isdir is not None:
                    try:
                        is_dir = bool(smart_isdir(p))
                    except Exception:
                        is_dir = False
                else:
                    is_dir = str(name).endswith("/")

                if is_dir:
                    stack.append(norm_root(p))
                else:
                    yield p
        return

    # 本地
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def smart_copy_compat(src: str, dst: str, overwrite: bool) -> None:
    """复制文件（本地<->S3 / S3<->S3 / 本地<->本地）"""
    # 只要涉及 s3:// 就优先用 megfile
    if (is_s3_path(src) or is_s3_path(dst)) and (smart_copy is not None):
        if (not overwrite) and exists_any(dst):
            return
        if is_s3_path(dst):
            ensure_parent_dir_smart(dst)
        else:
            ensure_parent_dir_local(dst)
        smart_copy(src, dst)
        return

    # 纯本地 fallback
    if is_s3_path(src) or is_s3_path(dst):
        raise RuntimeError("涉及 s3:// 但 smart_copy 不可用，请确认 megfile 安装/版本")

    if (not overwrite) and os.path.exists(dst):
        return
    ensure_parent_dir_local(dst)
    import shutil
    shutil.copy2(src, dst)


def make_matcher(ids: Set[str], mode: str):
    """
    mode:
      - stem: 文件名去扩展名 == model_id
      - filename: 文件名(含扩展名) == model_id
    """
    mode = mode.lower()
    if mode not in ("stem", "filename"):
        raise ValueError("match mode must be 'stem' or 'filename'")

    def _match(path: str) -> bool:
        base = os.path.basename(path)
        stem, _ext = os.path.splitext(base)
        if mode == "stem":
            # txt 里可能写了 xxx.png，这里也顺手兼容一下：
            # 如果 ids 里包含 base，则也算匹配
            return (stem in ids) or (base in ids)
        else:
            return base in ids

    return _match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_txt", required=True, help="每行一个 model_id 的 txt（本地或 s3://）")
    ap.add_argument("--src", required=True, help="源目录（本地或 s3://）")
    ap.add_argument("--dst", required=True, help="目标目录（本地或 s3://）")
    ap.add_argument("--match", default="stem", choices=["stem", "filename"],
                    help="匹配方式：stem=去扩展名；filename=含扩展名")
    ap.add_argument("--only_images", action="store_true", help="只处理图片扩展名（推荐开启）")
    ap.add_argument("--workers", type=int, default=64, help="并发复制线程数")
    ap.add_argument("--overwrite", action="store_true", help="目标已存在也覆盖")
    ap.add_argument("--dry_run", action="store_true", help="只统计不复制")
    ap.add_argument("--save_report", default="", help="保存报告 json（本地或 s3://，可选）")
    args = ap.parse_args()

    src_root = norm_root(args.src)
    dst_root = norm_root(args.dst)

    ids = load_ids_from_txt(args.ids_txt)
    if not ids:
        print("ids_txt 为空：没有任何 model_id", file=sys.stderr)
        return 2

    matcher = make_matcher(ids, args.match)

    matched_files: List[str] = []
    scanned = 0

    it = iter_all_files(src_root)
    if tqdm is not None:
        it = tqdm(it, desc="Scanning", unit="file")

    for p in it:
        scanned += 1
        if args.only_images and (not is_image_path(p)):
            continue
        if matcher(p):
            matched_files.append(p)

    # 统计每个 id 匹配次数（按 stem 统计更实用）
    id2count: Dict[str, int] = {k: 0 for k in ids}
    for p in matched_files:
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        # 根据你实际要的“等于 model_id”，一般用 stem
        if stem in id2count:
            id2count[stem] += 1
        if base in id2count:
            id2count[base] += 1

    missing = [k for k, c in id2count.items() if c == 0]

    report = {
        "ids_txt": args.ids_txt,
        "src_root": src_root,
        "dst_root": dst_root,
        "match_mode": args.match,
        "only_images": args.only_images,
        "ids_count": len(ids),
        "scanned_files": scanned,
        "matched_files": len(matched_files),
        "missing_ids_count": len(missing),
        "missing_ids_sample": sorted(missing)[:50],
        "top_match_sample": dict(list(sorted(id2count.items(), key=lambda x: -x[1]))[:50]),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.save_report:
        # 报告里写全量 id2count（可能大）
        full_report = {**report, "id2count": id2count}
        # 写本地或 s3：用 megfile smart_open
        data = json.dumps(full_report, ensure_ascii=False, indent=2).encode("utf-8")
        if is_s3_path(args.save_report):
            if mopen is None:
                raise RuntimeError("mopen 不可用，无法写入 s3:// report")
            ensure_parent_dir_smart(args.save_report)
            with mopen(args.save_report, "wb") as f:
                f.write(data)
        else:
            ensure_parent_dir_local(args.save_report)
            with open(args.save_report, "wb") as f:
                f.write(data)
        print(f"报告已保存: {args.save_report}")

    if args.dry_run:
        return 0

    # 复制：保持相对 src_root 的目录结构
    def _copy_task(src_path: str) -> str:
        rel = strip_prefix(src_path, src_root).replace("\\", "/")
        dst_path = norm_root(dst_root) + "/" + rel
        smart_copy_compat(src_path, dst_path, overwrite=args.overwrite)
        return dst_path

    ok = 0
    failed: List[Dict[str, str]] = []

    pbar = tqdm(total=len(matched_files), desc="Copying", unit="file") if (tqdm is not None) else None
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_copy_task, p): p for p in matched_files}
        for fut in as_completed(futs):
            src_p = futs[fut]
            try:
                _ = fut.result()
                ok += 1
            except Exception as e:
                failed.append({"src": src_p, "error": repr(e)})
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    print(f"完成：success={ok} failed={len(failed)}")
    if failed:
        print("失败样例（前20条）：")
        for item in failed[:20]:
            print(item)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

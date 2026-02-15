#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 keys.json 的 key（默认仅 value==1）从 S3 桶里递归找同名图片并拷贝到新目录，
同时保持原有目录嵌套结构（相对 src_root 的相对路径不变）。

示例：
python /data/LoraPipeline/utils/copy_from_json.py   --src s3://lanjinghong-data/loras_triplets/flux_0111_triplets   --dst s3://lanjinghong-data/loras_triplet/flux_0111_triplets_subset  --keys_json s3://lanjinghong-data/loras_triplets/flux_0111_triplets_all/judge_pos.json

python /data/LoraPipeline/utils/copy_from_json.py   --src s3://lanjinghong-data/loras_triplets/flux_0111_triplets   --dst s3://lanjinghong-data/loras_triplet/flux_0111_triplets_subset_neg  --keys_json s3://lanjinghong-data/loras_triplets/flux_0111_triplets_all/judge_neg.json   --keep_value 0

python /data/LoraPipeline/utils/copy_from_json.py   --src s3://lanjinghong-data/loras_triplets/flux_0111_triplets   --dst s3://lanjinghong-data/loras_triplet/flux_0111_triplets_subset_content_positive  --keys_json s3://lanjinghong-data/loras_triplets/flux_0111_triplets_all/judge_all_content.json   --keep_value 1


python /data/LoraPipeline/utils/copy_from_json.py   --src s3://lanjinghong-data/loras_triplets/flux_0111_triplets   --dst s3://lanjinghong-data/loras_triplet/flux_0111_triplets_subset_content_positive_dual  --keys_json s3://lanjinghong-data/loras_triplets/flux_0111_triplets_dual_judge/pos.json  --keep_value 1


也支持 dst 为 s3://...
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 keys.json 的 key（默认仅 value==1）从 S3 桶里递归找同名图片并拷贝到新目录，
同时保持原有目录嵌套结构（相对 src_root 的相对路径不变）。

示例：
python extract_by_keys_keep_tree.py \
  --src s3://lanjinghong-data/loras_triplets/flux_0111_triplets \
  --dst /data/selected_triplets/flux_0111_triplets_subset \
  --keys_json /path/to/keys.json \
  --workers 64

也支持 dst 为 s3://...
"""

import os
import sys
import json
import argparse
from megfile.smart import smart_open as mopen
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional, Set, Dict, List, Tuple

# ---------- optional deps ----------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------- megfile / local fallback ----------
try:
    from megfile.smart import (
        smart_listdir,
        smart_exists,
        smart_makedirs,
        smart_copy,
    )
    try:
        from megfile.smart import smart_isdir  # 有些版本有
    except Exception:
        smart_isdir = None  # type: ignore
except Exception as e:
    smart_listdir = None  # type: ignore
    smart_exists = None  # type: ignore
    smart_makedirs = None  # type: ignore
    smart_copy = None  # type: ignore
    smart_isdir = None  # type: ignore


def is_s3_path(p: str) -> bool:
    return p.startswith("s3://")


def norm_root(p: str) -> str:
    return p.rstrip("/")


def join_path(a: str, b: str) -> str:
    if a.endswith("/"):
        return a + b
    return a + "/" + b


def strip_prefix(path: str, prefix: str) -> str:
    """返回 path 相对 prefix 的相对路径（不以 / 开头）"""
    prefix_n = norm_root(prefix) + "/"
    if path == norm_root(prefix):
        return ""
    if not path.startswith(prefix_n):
        raise ValueError(f"path not under prefix: path={path} prefix={prefix}")
    return path[len(prefix_n):]


def ensure_parent_dir(dst_path: str) -> None:
    parent = os.path.dirname(dst_path)
    if not parent:
        return
    os.makedirs(parent, exist_ok=True)


def ensure_parent_dir_smart(dst_path: str) -> None:
    # smart_makedirs 需要目录路径
    parent = os.path.dirname(dst_path).replace("\\", "/")
    if not parent:
        return
    if smart_makedirs is None:
        raise RuntimeError("megfile.smart.smart_makedirs not available")
    smart_makedirs(parent)


def local_listdir(p: str) -> List[str]:
    return os.listdir(p)


def local_isdir(p: str) -> bool:
    return os.path.isdir(p)


def local_exists(p: str) -> bool:
    return os.path.exists(p)


def local_copy(src: str, dst: str, overwrite: bool) -> None:
    import shutil
    if (not overwrite) and os.path.exists(dst):
        return
    ensure_parent_dir(dst)
    shutil.copy2(src, dst)


def smart_copy_compat(src: str, dst: str, overwrite: bool) -> None:
    """
    megfile 的 smart_copy 一般会自动处理本地<->S3。
    若目的地已存在且 overwrite=False，则跳过。
    """
    if smart_exists is None or smart_copy is None:
        raise RuntimeError("megfile.smart not available, cannot copy S3 paths")

    if (not overwrite) and smart_exists(dst):
        return

    ensure_parent_dir_smart(dst)
    smart_copy(src, dst)


def iter_all_files(root: str) -> Iterable[str]:
    """
    递归遍历 root 下所有文件路径。
    优先使用 megfile.smart；否则走本地 os。
    """
    root = norm_root(root)

    # S3 / megfile 分支
    if is_s3_path(root):
        if smart_listdir is None:
            raise RuntimeError("Need megfile to traverse s3:// paths")

        stack = [root]
        while stack:
            cur = stack.pop()
            # smart_listdir 通常返回相对名字；目录可能带 /
            try:
                names = smart_listdir(cur)
            except Exception as e:
                # 某些情况下 cur 可能是文件而不是目录
                # 尝试当作文件处理：如果存在则 yield
                if smart_exists and smart_exists(cur):
                    yield cur
                continue

            for name in names:
                p = join_path(cur, name)

                # 判断是否目录：优先 smart_isdir；否则用 name 末尾 /
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

    # 本地分支
    stack = [root]
    while stack:
        cur = stack.pop()
        if not os.path.isdir(cur):
            if os.path.exists(cur):
                yield cur
            continue
        for name in os.listdir(cur):
            p = os.path.join(cur, name)
            if os.path.isdir(p):
                stack.append(p)
            else:
                yield p
def is_s3_path(p: str) -> bool:
    return p.startswith("s3://")

def read_json_any(path: str):
    """本地或 s3:// 都能读的 JSON"""
    if is_s3_path(path):
        if mopen is None:
            raise RuntimeError("mopen (megfile.smart.smart_open) 不可用，无法读取 s3:// JSON")
        with mopen(path, "rb") as f:
            raw = f.read()
    else:
        with open(path, "rb") as f:
            raw = f.read()

    text = raw.decode("utf-8-sig")  # 兼容 BOM
    return json.loads(text)

def load_wanted_keys(keys_json_path: str, keep_value):
    data = read_json_any(keys_json_path)
    if not isinstance(data, dict):
        raise ValueError("keys_json 必须是对象：{key: value, ...}")

    wanted = set()
    for k, v in data.items():
        if keep_value is None:
            wanted.add(str(k))
        else:
            try:
                if int(v) == int(keep_value):
                    wanted.add(str(k))
            except Exception:
                pass
    return wanted


def stem_of_path(p: str) -> str:
    base = os.path.basename(p)
    stem, _ = os.path.splitext(base)
    return stem


def copy_one(
    src_path: str,
    src_root: str,
    dst_root: str,
    overwrite: bool,
) -> Tuple[bool, str]:
    """
    复制单文件，返回 (copied_or_skipped_ok, dst_path)
    """
    rel = strip_prefix(src_path, src_root).replace("\\", "/")
    dst_path = norm_root(dst_root) + "/" + rel

    # 走 megfile（只要任一端是 s3:// 就用它）
    if is_s3_path(src_path) or is_s3_path(dst_path):
        smart_copy_compat(src_path, dst_path, overwrite=overwrite)
        return True, dst_path

    # 全本地
    local_copy(src_path, dst_path, overwrite=overwrite)
    return True, dst_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源根目录（s3://... 或本地路径）")
    ap.add_argument("--dst", required=True, help="目标根目录（s3://... 或本地路径）")
    ap.add_argument("--keys_json", required=True, help="包含 {key: 0/1} 的 json 文件路径")
    ap.add_argument("--keep_value", type=int, default=1, help="只保留 value==keep_value 的 key；设为 -1 表示不过滤")
    ap.add_argument("--workers", type=int, default=64, help="并发拷贝线程数")
    ap.add_argument("--overwrite", action="store_true", help="目标已存在时也覆盖")
    ap.add_argument("--dry_run", action="store_true", help="只统计不复制")
    ap.add_argument("--save_report", default="", help="保存报告 json 的路径（可选）")
    args = ap.parse_args()

    src_root = norm_root(args.src)
    dst_root = norm_root(args.dst)

    keep_value: Optional[int]
    if args.keep_value == -1:
        keep_value = None
    else:
        keep_value = args.keep_value

    wanted = load_wanted_keys(args.keys_json, keep_value=keep_value)
    if not wanted:
        print("No keys to extract (wanted set is empty).", file=sys.stderr)
        return 2

    # 扫描
    matched_files: List[str] = []
    missing_keys = set(wanted)

    it = iter_all_files(src_root)

    if tqdm is not None:
        it = tqdm(it, desc="Scanning", unit="file")

    for p in it:
        try:
            stem = stem_of_path(p)
        except Exception:
            continue
        if stem in wanted:
            matched_files.append(p)
            if stem in missing_keys:
                missing_keys.remove(stem)

    # 报告
    report = {
        "src_root": src_root,
        "dst_root": dst_root,
        "wanted_keys_count": len(wanted),
        "matched_files_count": len(matched_files),
        "missing_keys_count": len(missing_keys),
        "missing_keys_sample": sorted(list(missing_keys))[:50],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    # 并发拷贝
    ok = 0
    failed: List[Dict[str, str]] = []

    pbar = tqdm(total=len(matched_files), desc="Copying", unit="file") if (tqdm is not None) else None

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {
            ex.submit(copy_one, src_path, src_root, dst_root, args.overwrite): src_path
            for src_path in matched_files
        }
        for fut in as_completed(futs):
            src_path = futs[fut]
            try:
                _, dst_path = fut.result()
                ok += 1
            except Exception as e:
                failed.append({"src": src_path, "error": repr(e)})
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    print(f"Done. success={ok} failed={len(failed)}")

    if args.save_report:
        out = {
            **report,
            "success": ok,
            "failed": failed[:200],  # 防止太大
        }
        # 保存到本地文件（更通用）
        ensure_parent_dir(args.save_report)
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Report saved to: {args.save_report}")

    # 额外：把 missing_keys 也落个文件，方便你二次排查
    if args.save_report:
        miss_path = os.path.splitext(args.save_report)[0] + ".missing_keys.json"
        with open(miss_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(missing_keys)), f, ensure_ascii=False, indent=2)
        print(f"Missing keys saved to: {miss_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

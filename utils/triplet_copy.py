#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 combine_root 的 content_id__style_id 组合目录中取“融合图”，
并从 demo_root/<model_id>/demo_images 中把所有图片都用上：

输出到 out_root：
  - style_and_content/    : 融合图
  - style_1/, style_2/... : style 的第1/2/...张 demo 图
  - content_1/, content_2/... : content 的第1/2/...张 demo 图

同一个融合图对应的所有拷贝文件名统一为：
  style_id__content_id_<融合图原文件名>
"""

import os
import sys
import json
import re
import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from megfile.smart import (
    smart_listdir,
    smart_exists,
    smart_makedirs,
    smart_copy,
    smart_open as mopen,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def norm_dir(p: str) -> str:
    return p if p.endswith("/") else (p + "/")


def join_path(base: str, name: str) -> str:
    return base + name if base.endswith("/") else base + "/" + name


def ensure_dir(d: str) -> str:
    d = norm_dir(d)
    try:
        smart_makedirs(d, exist_ok=True)
    except TypeError:
        try:
            smart_makedirs(d)
        except Exception as e:
            if "File exists" in str(e):
                return d
            raise
    except Exception as e:
        if "File exists" in str(e):
            return d
        raise
    return d


def list_images_one_level(dir_path: str) -> List[str]:
    """只扫一层，返回该目录下所有图片的完整路径（排序后）"""
    dir_path = norm_dir(dir_path)
    out: List[str] = []
    try:
        items = smart_listdir(dir_path)
    except Exception as e:
        print(f"[WARN] listdir failed: {dir_path} | {e}", file=sys.stderr)
        return out

    for it in items:
        if it.endswith("/"):
            continue
        if is_image(it):
            out.append(join_path(dir_path, it))

    out.sort(key=lambda x: os.path.basename(x))
    return out


def list_images(dir_path: str, recursive: bool = False) -> List[str]:
    """
    返回 dir_path 下所有图片完整路径。
    - recursive=False：只扫一层
    - recursive=True ：递归扫子目录
    """
    dir_path = norm_dir(dir_path)
    out: List[str] = []
    if not recursive:
        return list_images_one_level(dir_path)

    stack = [dir_path]
    while stack:
        cur = stack.pop()
        try:
            items = smart_listdir(cur)
        except Exception as e:
            print(f"[WARN] listdir failed: {cur} | {e}", file=sys.stderr)
            continue

        for it in items:
            if it.endswith("/"):
                stack.append(join_path(cur, it))
                continue
            if is_image(it):
                out.append(join_path(cur, it))

    out.sort(key=lambda x: os.path.basename(x))
    return out


def count_pair_images(dir_path: str, prefix: str) -> int:
    try:
        imgs = list_images_one_level(dir_path)
    except Exception:
        return 0
    cnt = 0
    for p in imgs:
        if os.path.basename(p).startswith(prefix):
            cnt += 1
    return cnt


def count_pair_json(dir_path: str, prefix: str) -> int:
    try:
        items = smart_listdir(norm_dir(dir_path))
    except Exception:
        return 0
    cnt = 0
    for it in items:
        if it.endswith("/"):
            continue
        if it.startswith(prefix) and it.lower().endswith(".json"):
            cnt += 1
    return cnt


def parse_pair_dirname(dirname: str) -> Optional[Tuple[str, str]]:
    """从 'content_id__style_id/' 解析 (content_id, style_id)"""
    d = dirname.rstrip("/")
    if "__" not in d:
        return None
    content_id, style_id = d.split("__", 1)
    content_id, style_id = content_id.strip(), style_id.strip()
    if not content_id or not style_id:
        return None
    return content_id, style_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine_root", required=True, help="组合图根目录：包含 content_id__style_id/ 子目录")
    ap.add_argument("--demo_root", required=True, help="单 LoRA 根目录：包含 <model_id>/demo_images/")
    ap.add_argument("--out_root", required=True, help="输出根目录")
    ap.add_argument("--recursive", action="store_true", help="是否递归扫描组合图子目录下的图片")
    ap.add_argument("--max_pairs", type=int, default=0, help="最多处理多少个 pair（0=不限制）")
    ap.add_argument("--max_images_per_pair", type=int, default=0, help="每个 pair 最多处理多少张融合图（0=不限制）")
    ap.add_argument("--sample_images_per_pair", type=int, default=0, help="每个 pair 随机抽取多少张融合图（0=不限制）")
    ap.add_argument("--workers", type=int, default=8, help="并发扫描/拷贝线程数")
    ap.add_argument("--overwrite", action="store_true", help="目标存在时是否覆盖")
    ap.add_argument("--dry_run", action="store_true", help="只打印不拷贝")
    ap.add_argument("--output_prompt_json", default=None, help="输出 prompt 汇总 JSON 路径（本地或桶）；不传则不写")
    ap.add_argument("--pair-ids", default=None, help="仅处理指定的组合，支持 txt 或 json（json 取 key）；提取前两个数字作为 content/style")
    ap.add_argument("--content_ids_txt", default=None, help="content 白名单 txt，每行一个 model_id")
    ap.add_argument("--style_ids_txt", default=None, help="style 白名单 txt，每行一个 model_id")
    args = ap.parse_args()

    combine_root = norm_dir(args.combine_root)
    demo_root = args.demo_root.rstrip("/")
    out_root = args.out_root.rstrip("/")

    print(f"[INFO] combine_root={combine_root} demo_root={demo_root} out_root={out_root}")
    out_style_and_content = ensure_dir(f"{out_root}/style_and_content")
    out_prompt = ensure_dir(f"{out_root}/prompt")

    print(f"[INFO] workers={args.workers}")
    demo_cache: Dict[str, List[str]] = {}
    demo_lock = threading.Lock()
    prompts_map: Dict[str, str] = {}
    prompts_lock = threading.Lock()

    def get_demo_images(model_id: str) -> List[str]:
        with demo_lock:
            cached = demo_cache.get(model_id)
        if cached is not None:
            return cached

        direct_dir = f"{demo_root}/{model_id}/"
        imgs: List[str] = []

        if smart_exists(direct_dir):
            imgs = list_images_one_level(direct_dir)

        if not imgs:
            legacy_dir = f"{demo_root}/{model_id}/demo_images/"
            if smart_exists(legacy_dir):
                imgs = list_images_one_level(legacy_dir)

        with demo_lock:
            demo_cache[model_id] = imgs
        return imgs

    def read_pair_list(path: Optional[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if not path or not smart_exists(path):
            return pairs

        def add_from_str(s: str):
            s = s.strip()
            if not s or s.startswith("#"):
                return
            nums = re.findall(r"\d+", s)
            if len(nums) >= 2:
                sid, cid = nums[0], nums[1]
                pairs.append((sid, cid))

        try:
            if str(path).lower().endswith(".json"):
                with mopen(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    for k in obj.keys():
                        add_from_str(str(k))
                elif isinstance(obj, list):
                    for it in obj:
                        add_from_str(str(it))
                else:
                    add_from_str(str(obj))
            else:
                with mopen(path, "r", encoding="utf-8") as f:
                    for line in f:
                        add_from_str(line)
        except Exception as e:
            print(f"[ERR] Read {path} failed: {e}", file=sys.stderr)
        return pairs

    def read_id_set(path: Optional[str]) -> Optional[set]:
        if not path:
            return None
        if not smart_exists(path):
            print(f"[WARN] whitelist not found: {path}")
            return None
        ids: set = set()
        try:
            with mopen(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    ids.add(s)
        except Exception as e:
            print(f"[ERR] Read {path} failed: {e}", file=sys.stderr)
            return None
        return ids

    def has_images(dir_path: str) -> bool:
        try:
            imgs = list_images(dir_path, recursive=args.recursive)
        except Exception:
            return False
        return len(imgs) > 0

    pair_dirs: List[str] = []
    content_whitelist = read_id_set(args.content_ids_txt) if args.content_ids_txt else None
    style_whitelist = read_id_set(args.style_ids_txt) if args.style_ids_txt else None

    if content_whitelist is not None and style_whitelist is not None:
        content_list = sorted(content_whitelist)
        style_list = sorted(style_whitelist)
        for cid in content_list:
            for sid in style_list:
                dname = f"{cid}__{sid}/"
                full = join_path(combine_root, dname)
                if smart_exists(full):
                    pair_dirs.append(dname)
        print(f"[INFO] whitelist pairs prepared: content={len(content_whitelist)} style={len(style_whitelist)}")
    elif args.pair_ids:
        wanted = read_pair_list(args.pair_ids)
        print(f"[INFO] pair-ids provided: {len(wanted)} lines")
        for sid, cid in wanted:
            dname = f"{cid}__{sid}/"
            full = join_path(combine_root, dname)
            if smart_exists(full):
                pair_dirs.append(dname)
            else:
                print(f"[WARN] missing pair dir: {full}")
        pair_dirs.sort()
    else:
        try:
            print(f"[INFO] scanning pair dirs under: {combine_root}")
            pair_entries = smart_listdir(combine_root)
        except Exception as e:
            print(f"[ERROR] listdir combine_root failed: {combine_root} | {e}", file=sys.stderr)
            sys.exit(2)

        normalized_pairs: List[str] = []
        for p in pair_entries:
            name = p.rstrip("/")
            if "__" not in name:
                continue
            normalized_pairs.append(name + "/")

        pair_dirs = sorted(normalized_pairs)

    filtered_dirs: List[str] = []
    for pd in pair_dirs:
        pair_path = join_path(combine_root, pd)
        if has_images(pair_path):
            filtered_dirs.append(pd)
        else:
            print(f"[WARN] empty pair dir: {pair_path}")
    pair_dirs = filtered_dirs

    if args.max_pairs and len(pair_dirs) > args.max_pairs:
        pair_dirs = pair_dirs[:args.max_pairs]
    print(f"[INFO] total pairs={len(pair_dirs)}")
    
    processed_pairs = 0
    copied = 0
    skipped = 0

    def process_pair(idx_pair: int, pd: str) -> Tuple[int, int, int]:
        parsed = parse_pair_dirname(pd)
        if not parsed:
            return 0, 0, 0
        content_id, style_id = parsed

        pair_path = join_path(combine_root, pd)
        print(f"[INFO] pair {idx_pair}/{len(pair_dirs)} -> {pair_path}")
        fusion_imgs = list_images(pair_path, recursive=args.recursive)
        print(f"[INFO] fusion images={len(fusion_imgs)} recursive={args.recursive}")
        if not fusion_imgs:
            print(f"[WARN] empty pair dir: {pair_path}")
            return 0, 0, 0

        if args.sample_images_per_pair:
            target = args.sample_images_per_pair
        else:
            target = args.max_images_per_pair

        if target and len(fusion_imgs) > target:
            if args.sample_images_per_pair:
                fusion_imgs = random.sample(fusion_imgs, target)
            else:
                fusion_imgs = fusion_imgs[:target]
            print(f"[INFO] fusion images after limit={len(fusion_imgs)} sample={bool(args.sample_images_per_pair)}")

        style_demos = get_demo_images(style_id)
        content_demos = get_demo_images(content_id)
        if not style_demos or not content_demos:
            print(f"[WARN] missing demo_images: style={style_id}({len(style_demos)}) content={content_id}({len(content_demos)}) -> skip pair {pd}")
            return 0, 0, 0
        print(f"[INFO] style demos={len(style_demos)} content demos={len(content_demos)}")

        prefix = f"{style_id}__{content_id}_"
        expected = len(fusion_imgs)
        if not args.overwrite and expected:
            complete = True
            if count_pair_images(out_style_and_content, prefix) < expected:
                complete = False
            if complete:
                for i in range(1, len(style_demos) + 1):
                    d = f"{out_root}/style_{i}"
                    if not smart_exists(d) or count_pair_images(d, prefix) < expected:
                        complete = False
                        break
            if complete:
                for i in range(1, len(content_demos) + 1):
                    d = f"{out_root}/content_{i}"
                    if not smart_exists(d) or count_pair_images(d, prefix) < expected:
                        complete = False
                        break
            if complete:
                if not smart_exists(out_prompt) or count_pair_json(out_prompt, prefix) < expected:
                    complete = False
            if complete:
                print(f"[INFO] pair exists and complete, skip: {content_id}__{style_id} expected={expected}")
                return 1, 0, 0

        local_copied = 0
        local_skipped = 0
        # 为本 pair 创建 style_1..style_N / content_1..content_M 目录（按 demo 数量）
        style_out_dirs = []
        for i in range(1, len(style_demos) + 1):
            style_out_dirs.append(ensure_dir(f"{out_root}/style_{i}"))

        content_out_dirs = []
        for i in range(1, len(content_demos) + 1):
            content_out_dirs.append(ensure_dir(f"{out_root}/content_{i}"))

        for fusion_path in fusion_imgs:
            orig_name = os.path.basename(fusion_path)
            out_name = f"{style_id}__{content_id}_{orig_name}"

            # 1) 融合图
            dst_fusion = join_path(out_style_and_content, out_name)
            if (not args.overwrite) and smart_exists(dst_fusion):
                pass
            else:
                if args.dry_run:
                    print("[DRYRUN] fusion  :", fusion_path, "->", dst_fusion)
                else:
                    try:
                        smart_copy(fusion_path, dst_fusion)
                        local_copied += 1
                    except Exception as e:
                        local_skipped += 1
                        print(f"[WARN] copy fusion failed: {dst_fusion} | {e}", file=sys.stderr)

            # 2) style：所有 demo_images 都用上 -> style_1/style_2/...
            for idx, src in enumerate(style_demos, start=1):
                dst = join_path(style_out_dirs[idx - 1], out_name)
                if (not args.overwrite) and smart_exists(dst):
                    continue
                if args.dry_run:
                    print(f"[DRYRUN] style_{idx} :", src, "->", dst)
                else:
                    try:
                        smart_copy(src, dst)
                        local_copied += 1
                    except Exception as e:
                        local_skipped += 1
                        print(f"[WARN] copy style_{idx} failed: {dst} | {e}", file=sys.stderr)

            for idx, src in enumerate(content_demos, start=1):
                dst = join_path(content_out_dirs[idx - 1], out_name)
                if (not args.overwrite) and smart_exists(dst):
                    continue
                if args.dry_run:
                    print(f"[DRYRUN] content_{idx} :", src, "->", dst)
                else:
                    try:
                        smart_copy(src, dst)
                        local_copied += 1
                    except Exception as e:
                        local_skipped += 1
                        print(f"[WARN] copy content_{idx} failed: {dst} | {e}", file=sys.stderr)
            meta_stem = os.path.splitext(orig_name)[0]
            meta_path = join_path(pair_path, meta_stem + ".json")
            prompt_value = ""
            if smart_exists(meta_path):
                try:
                    with mopen(meta_path, "r", encoding="utf-8") as f:
                        meta_obj = json.load(f)
                    ptxt = meta_obj.get("prompt")
                    if isinstance(ptxt, str):
                        prompt_value = ptxt
                except Exception as e:
                    print(f"[WARN] read meta json failed: {meta_path} | {e}", file=sys.stderr)
            if args.dry_run:
                print(f"[DRYRUN] prompt :", meta_path, "->", join_path(out_prompt, os.path.splitext(out_name)[0] + '.json'))
            else:
                prompt_dst = join_path(out_prompt, os.path.splitext(out_name)[0] + ".json")
                if args.overwrite or (not smart_exists(prompt_dst)):
                    try:
                        with mopen(prompt_dst, "w", encoding="utf-8") as f:
                            json.dump({"prompt": prompt_value}, f, ensure_ascii=False, indent=2)
                        local_copied += 1
                    except Exception as e:
                        local_skipped += 1
                        print(f"[WARN] write prompt json failed: {prompt_dst} | {e}", file=sys.stderr)
            if args.output_prompt_json and prompt_value:
                with prompts_lock:
                    prompts_map[out_name] = prompt_value
        return 1, local_copied, local_skipped

    if args.workers <= 1:
        for idx_pair, pd in enumerate(pair_dirs, start=1):
            p, c, s = process_pair(idx_pair, pd)
            processed_pairs += p
            copied += c
            skipped += s
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_pair, idx_pair, pd)
                for idx_pair, pd in enumerate(pair_dirs, start=1)
            ]
            for fut in as_completed(futures):
                try:
                    p, c, s = fut.result()
                except Exception as e:
                    skipped += 1
                    print(f"[WARN] pair task failed: {e}", file=sys.stderr)
                    continue
                processed_pairs += p
                copied += c
                skipped += s

    print(f"[DONE] pairs={processed_pairs} copied={copied} skipped={skipped}")
    if args.output_prompt_json:
        out_json_path = args.output_prompt_json
        out_dir = os.path.dirname(out_json_path)
        if out_dir:
            try:
                ensure_dir(out_dir)
            except Exception as e:
                print(f"[WARN] ensure dir for prompts json failed: {out_dir} | {e}", file=sys.stderr)
        try:
            with mopen(out_json_path, "w", encoding="utf-8") as f:
                json.dump(prompts_map, f, ensure_ascii=False, indent=2)
            print(f"[OUT] prompts json -> {out_json_path}")
        except Exception as e:
            print(f"[WARN] write prompts json failed: {out_json_path} | {e}", file=sys.stderr)
    print(f"[OUT]  {out_root}/style_and_content/  {out_root}/style_*/  {out_root}/content_*/  {out_root}/prompt/")


if __name__ == "__main__":
    main()

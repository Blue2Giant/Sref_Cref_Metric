#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import json
import math
import random
import argparse
from typing import Dict, Any, List, Tuple, Optional, Set

from tqdm import tqdm
from PIL import Image, ImageOps

from megfile.smart import (
    smart_listdir,
    smart_exists,
    smart_makedirs,
    smart_open as mopen,
    smart_copy as mcopy,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ------------------------- path helpers -------------------------
def s3_join(a: str, b: str) -> str:
    a = a.rstrip("/")
    b = b.lstrip("/")
    return f"{a}/{b}"


def ensure_dir(p: str) -> None:
    """
    megfile/某些S3网关会对“目录marker对象”比较敏感，重复创建目录有时会抛 FileExistsError。
    这里做成幂等：已存在就忽略。
    """
    try:
        smart_makedirs(p.rstrip("/") + "/")
    except FileExistsError:
        pass


def is_image_path(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return (ext in IMG_EXTS) or (ext == "")


# ------------------------- bins -------------------------
def decimals_for_step(step: float) -> int:
    if step <= 0:
        return 3
    s = f"{step:.10f}".rstrip("0").rstrip(".")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def fmt_num(x: float, dec: int) -> str:
    if dec <= 0:
        return str(int(round(x)))
    s = f"{x:.{dec}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def make_bins_from_width(min_score: float, max_score: float, bin_width: float) -> List[Tuple[float, float]]:
    if bin_width <= 0:
        raise ValueError("bin_width 必须 > 0")
    n = int(math.ceil((max_score - min_score) / bin_width))
    bins = []
    for i in range(n):
        lo = min_score + i * bin_width
        hi = min(lo + bin_width, max_score)
        bins.append((lo, hi))
    return bins


def make_bins_from_edges(edges: List[float]) -> List[Tuple[float, float]]:
    if len(edges) < 2:
        raise ValueError("bins-edges 至少给2个边界")
    e = sorted(float(x) for x in edges)
    bins = []
    for i in range(len(e) - 1):
        bins.append((e[i], e[i + 1]))
    return bins


def bin_index(score: float, bins: List[Tuple[float, float]]) -> Optional[int]:
    # lo <= s < hi，最后一个桶允许 s == hi
    for i, (lo, hi) in enumerate(bins):
        if i == len(bins) - 1:
            if lo <= score <= hi:
                return i
        else:
            if lo <= score < hi:
                return i
    return None


def bin_label(lo: float, hi: float, dec: int) -> str:
    return f"{fmt_num(lo, dec)}-{fmt_num(hi, dec)}"


# ------------------------- JSON robust loader -------------------------
def load_json_smart(path: str) -> Dict[str, Any]:
    with mopen(path, "r") as f:
        txt = f.read()

    # strict json
    try:
        return json.loads(txt)
    except Exception:
        pass

    # tolerate Chinese punctuation & trailing commas
    t = txt.replace("，", ",").replace("：", ":")
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return json.loads(t)


# ------------------------- model scanning -------------------------
def list_model_dirs(root: str) -> List[str]:
    if not smart_exists(root):
        raise FileNotFoundError(f"root 不存在: {root}")
    names = smart_listdir(root)
    out = []
    for n in names:
        n = n.rstrip("/")
        if not n:
            continue
        out.append(s3_join(root, n))
    return out


def get_demo_dir_from_json(model_dir: str, data: Dict[str, Any]) -> str:
    for key in ("demo_dir", "probe_dir"):
        v = data.get(key)
        if isinstance(v, str) and v:
            return v.rstrip("/")
    # fallback
    p = s3_join(model_dir, "demo_images")
    return p


def list_demo_images(demo_dir: str) -> List[str]:
    if not demo_dir or (not smart_exists(demo_dir)):
        return []
    out = []
    for n in smart_listdir(demo_dir):
        n = n.rstrip("/")
        if not n:
            continue
        p = s3_join(demo_dir, n)
        if is_image_path(p):
            out.append(p)
    return out


# ------------------------- image IO (remote) -------------------------
def read_image_smart(uri: str) -> Image.Image:
    with mopen(uri, "rb") as f:
        b = f.read()
    im = Image.open(io.BytesIO(b))
    im.load()
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    return im


def save_png_smart(im: Image.Image, uri: str) -> None:
    ensure_dir(os.path.dirname(uri))
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    bio.seek(0)
    with mopen(uri, "wb") as f:
        f.write(bio.read())


def copy_to_png(src: str, dst_png: str) -> None:
    """
    统一输出成 .png（你的命名要求固定 0.png / 1.png ...）
    - src 如果是 png：直接 mcopy
    - 否则：读入转 png 写出
    """
    ext = os.path.splitext(src)[1].lower()
    if ext == ".png":
        ensure_dir(os.path.dirname(dst_png))
        # 有些网关对重复写非常敏感，这里尽量避免“覆盖”，所以配合“选 unused idx”使用
        mcopy(src, dst_png)
        return

    im = read_image_smart(src)
    if im.mode != "RGB":
        im = im.convert("RGB")
    save_png_smart(im, dst_png)


# ------------------------- ref collage -------------------------
def pick_ref_uris(demo_imgs: List[str], k: int, rng: random.Random) -> List[str]:
    if k <= 0 or not demo_imgs:
        return []
    if len(demo_imgs) <= k:
        return list(demo_imgs)
    return rng.sample(demo_imgs, k)


def make_collage(ref_uris: List[str], cell_size: int = 384) -> Optional[Image.Image]:
    """
    K=4 -> 2x2 四宫格
    其他 K -> 尽量方形网格
    统一用 ImageOps.fit 做居中裁切到 cell_size（可保证整齐）
    """
    if not ref_uris:
        return None

    ims: List[Image.Image] = []
    for u in ref_uris:
        try:
            im = read_image_smart(u)
            if im.mode != "RGB":
                im = im.convert("RGB")
            ims.append(im)
        except Exception:
            continue
    if not ims:
        return None

    k = len(ims)
    if k == 4:
        rows, cols = 2, 2
    elif k <= 3:
        rows, cols = 1, k
    else:
        cols = int(math.ceil(math.sqrt(k)))
        rows = int(math.ceil(k / cols))

    fitted = [
        ImageOps.fit(im, (cell_size, cell_size), method=Image.BICUBIC, centering=(0.5, 0.5))
        for im in ims
    ]

    canvas = Image.new("RGB", (cols * cell_size, rows * cell_size), (255, 255, 255))
    for idx, im in enumerate(fitted):
        r = idx // cols
        c = idx % cols
        canvas.paste(im, (c * cell_size, r * cell_size))

    return canvas


# ------------------------- resume support -------------------------
_RE_IDX = re.compile(r"^(\d+)\.png$", re.IGNORECASE)


def scan_used_small_indices(bin_dir: str, target_n: int) -> Set[int]:
    """
    只关心 0..target_n-1 内哪些已经存在，方便断点续跑“补空缺”
    """
    used: Set[int] = set()
    if not smart_exists(bin_dir):
        return used
    try:
        names = smart_listdir(bin_dir)
    except Exception:
        return used
    for n in names:
        n = n.rstrip("/")
        m = _RE_IDX.match(n)
        if not m:
            continue
        idx = int(m.group(1))
        if 0 <= idx < target_n:
            used.add(idx)
    return used


def pick_next_unused(used: Set[int], target_n: int) -> Optional[int]:
    for i in range(target_n):
        if i not in used:
            return i
    return None


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser("Global bin sampler (stop when each bin filled) + per-sample ref collage.")
    ap.add_argument("--root", required=True, help="根目录：下面是很多 model_id/ (s3://.../loras_eval_flux)")
    ap.add_argument("--out", required=True, help="输出目录（不按 model_id 分层）")

    # bins
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-score", type=float, default=1.0)
    ap.add_argument("--bin-width", type=float, default=0.2, help="例如 0.2 -> 0-0.2,0.2-0.4,...")
    ap.add_argument("--bins-edges", nargs="*", type=float, default=None,
                    help="显式边界：如 0 0.2 0.4 0.6 0.8 1.0（给了就忽略 bin-width）")

    # fill rule
    ap.add_argument("--n-per-bin", type=int, default=10, help="每个 bin 需要凑齐 N 张")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-name", default="vlm_content_similarity.json")

    # refs
    ap.add_argument("--refs-per-sample", type=int, default=4, help="每个样本从 demo_images 随机取 K 张做拼图")
    ap.add_argument("--cell-size", type=int, default=384, help="ref拼图每格尺寸（像素），默认 384")
    ap.add_argument("--no-ref", action="store_true", help="只采样 eval 图，不生成 ref 拼图")

    # io
    ap.add_argument("--force-png", action="store_true", default=True,
                    help="强制输出为 png（固定命名 0.png/1.png...），默认开启")
    ap.add_argument("--max-models", type=int, default=0, help="调试用：最多处理多少个 model，0 不限制")

    args = ap.parse_args()

    root = args.root.rstrip("/")
    out_root = args.out.rstrip("/")
    ensure_dir(out_root)

    rng = random.Random(args.seed)

    # build bins
    if args.bins_edges and len(args.bins_edges) >= 2:
        bins = make_bins_from_edges(args.bins_edges)
        # 估算小数位仅用于目录名更美观
        steps = [abs(hi - lo) for (lo, hi) in bins if hi > lo]
        step = min(steps) if steps else 0.1
        dec = decimals_for_step(step)
    else:
        bins = make_bins_from_width(args.min_score, args.max_score, args.bin_width)
        dec = decimals_for_step(args.bin_width)

    bin_names = [bin_label(lo, hi, dec) for (lo, hi) in bins]
    bin_img_dirs = [s3_join(out_root, name) for name in bin_names]
    bin_ref_dirs = [s3_join(out_root, f"{name}ref") for name in bin_names]

    # create dirs (idempotent)
    for d in bin_img_dirs:
        ensure_dir(d)
    if not args.no_ref:
        for d in bin_ref_dirs:
            ensure_dir(d)

    target = int(args.n_per_bin)
    if target <= 0:
        raise ValueError("--n-per-bin 必须 > 0")

    # resume: scan used indices in each bin
    used_small: List[Set[int]] = []
    for d in bin_img_dirs:
        used_small.append(scan_used_small_indices(d, target))

    def bin_full(i: int) -> bool:
        return len(used_small[i]) >= target

    def all_full() -> bool:
        return all(bin_full(i) for i in range(len(bins)))

    # manifest for traceability
    manifest: Dict[str, Any] = {
        "root": root,
        "out": out_root,
        "bins": [{"name": n, "lo": float(b[0]), "hi": float(b[1]), "target": target} for n, b in zip(bin_names, bins)],
        "refs_per_sample": int(args.refs_per_sample),
        "samples": {n: [] for n in bin_names},
        "resume_used": {n: sorted(list(used_small[i])) for i, n in enumerate(bin_names)},
    }

    model_dirs = list_model_dirs(root)
    if args.max_models and args.max_models > 0:
        model_dirs = model_dirs[: args.max_models]

    print(f"[INFO] bins={bin_names} target={target} (resume counts={[len(s) for s in used_small]})")
    print(f"[INFO] model_dirs={len(model_dirs)}")

    # scan models
    for model_dir in tqdm(model_dirs, desc="scan models"):
        if all_full():
            break

        json_path = s3_join(model_dir, args.json_name)
        if not smart_exists(json_path):
            continue

        try:
            data = load_json_smart(json_path)
        except Exception as e:
            print(f"[WARN] JSON解析失败: {json_path} | {e}")
            continue

        per_eval = data.get("per_eval_similarity", {}) or {}
        if not isinstance(per_eval, dict) or not per_eval:
            continue

        demo_dir = get_demo_dir_from_json(model_dir, data)
        demo_imgs = [] if args.no_ref else list_demo_images(demo_dir)

        # iterate images in this model
        for img_uri, score_val in per_eval.items():
            if all_full():
                break

            try:
                s = float(score_val)
            except Exception:
                continue

            bi = bin_index(s, bins)
            if bi is None:
                continue
            if bin_full(bi):
                continue

            idx = pick_next_unused(used_small[bi], target)
            if idx is None:
                continue  # already full

            dst_img = s3_join(bin_img_dirs[bi], f"{idx}.png")
            dst_ref = s3_join(bin_ref_dirs[bi], f"{idx}.png") if not args.no_ref else ""

            # copy eval -> dst
            try:
                if args.force_png:
                    copy_to_png(img_uri, dst_img)
                else:
                    ensure_dir(os.path.dirname(dst_img))
                    mcopy(img_uri, dst_img)
            except FileExistsError:
                # 极端情况下：并发/历史遗留导致同名已存在，则标记已用并跳过
                used_small[bi].add(idx)
                continue
            except Exception as e:
                print(f"[WARN] copy eval 失败: {img_uri} -> {dst_img} | {e}")
                continue

            # make ref collage
            if (not args.no_ref) and args.refs_per_sample > 0:
                try:
                    ref_uris = pick_ref_uris(demo_imgs, args.refs_per_sample, rng)
                    collage = make_collage(ref_uris, cell_size=int(args.cell_size))
                    if collage is not None:
                        save_png_smart(collage, dst_ref)
                except Exception as e:
                    print(f"[WARN] ref拼图失败: model={model_dir} demo_dir={demo_dir} | {e}")

            used_small[bi].add(idx)

            bin_name = bin_names[bi]
            manifest["samples"][bin_name].append({
                "idx": idx,
                "score": s,
                "src_image": img_uri,
                "dst_image": dst_img,
                "dst_ref": dst_ref,
                "model_dir": model_dir,
                "demo_dir": demo_dir,
            })

    # write manifest
    manifest["filled"] = {bin_names[i]: len(used_small[i]) for i in range(len(bins))}
    manifest["all_full"] = all_full()

    manifest_path = s3_join(out_root, "manifest.json")
    with mopen(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print("filled:", manifest["filled"])
    print("all_full:", manifest["all_full"])
    print("manifest:", manifest_path)
    if not manifest["all_full"]:
        print("[WARN] 未全部凑齐：说明某些 bin 在全量数据里样本不足，或你限制了 max-models。")


if __name__ == "__main__":
    main()

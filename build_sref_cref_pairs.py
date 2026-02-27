import argparse
import json
import os
import random
from io import BytesIO
from typing import List, Tuple, Dict, Optional

from PIL import Image
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from megfile.smart import smart_listdir, smart_exists, smart_makedirs, smart_open as mopen
"""
python /data/LoraPipeline/utils/build_sref_cref_pairs.py \
  --content-dir /mnt/jfs/bench-bucket/sref_bench/bench_1106_content_prompt \
  --style-dir /mnt/jfs/bench-bucket/sref_bench/bench_0222_style/bench_1022_style/ \
  --out-dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref \
  --num-combos 800 \
  --max-side 2048 \
  --seed 42
python /data/LoraPipeline/utils/build_sref_cref_pairs.py \
  --content-dir /mnt/jfs/bench-bucket/sref_bench/bench_1106_content_prompt \
  --style-dir /mnt/jfs/bench-bucket/sref_bench/bench_0222_style/bench_1022_style/ \
  --out-dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_sref \
  --num-combos 800 \
  --seed 200 \
  --max-side 2048 
"""

def norm_dir(p: str) -> str:
    return p if p.endswith("/") else (p + "/")


def join_path(base: str, name: str) -> str:
    return base + name if base.endswith("/") else base + "/" + name


def list_images(dir_path: str, exts: List[str]) -> List[str]:
    dir_path = norm_dir(dir_path)
    try:
        items = smart_listdir(dir_path)
    except Exception:
        return []
    out: List[str] = []
    for it in items:
        name = os.path.basename(str(it).rstrip("/"))
        low = name.lower()
        if any(low.endswith(x) for x in exts):
            out.append(join_path(dir_path, name))
    out.sort(key=lambda x: os.path.basename(x))
    return out


def read_edit_prompts(img_path: str) -> Optional[List[str]]:
    base = os.path.splitext(os.path.basename(img_path))[0]
    parent = os.path.dirname(img_path)
    if not parent:
        parent = "."
    json_path = join_path(parent, base + ".json")
    if not smart_exists(json_path):
        return None
    try:
        with mopen(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    prompts = obj.get("edit_prompts")
    cleaned: List[str] = []
    if isinstance(prompts, list):
        cleaned = [p for p in prompts if isinstance(p, str) and p.strip()]
    elif isinstance(prompts, dict):
        items = sorted(prompts.items(), key=lambda x: str(x[0]))
        cleaned = [v for _, v in items if isinstance(v, str) and v.strip()]
    else:
        return None
    return cleaned if cleaned else None


def read_bytes(path: str) -> Optional[bytes]:
    try:
        with mopen(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def save_png(src_path: str, dst_path: str, max_side: int) -> bool:
    data = read_bytes(src_path)
    if not data:
        return False
    try:
        img = Image.open(BytesIO(data))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        w, h = img.size
        side = max(w, h)
        if max_side > 0 and side > max_side:
            scale = max_side / float(side)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        with mopen(dst_path, "wb") as f:
            f.write(buf.getvalue())
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content-dir", required=True)
    ap.add_argument("--style-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-combos", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", default=".png,.jpg,.jpeg,.webp,.bmp,.avif")
    ap.add_argument("--max-side", type=int, default=2048)
    args = ap.parse_args()

    exts = [x.strip().lower() for x in args.exts.split(",") if x.strip()]
    content_paths = list_images(args.content_dir, exts)
    style_paths = list_images(args.style_dir, exts)

    content_items: List[Tuple[str, List[str]]] = []
    for p in content_paths:
        prompts = read_edit_prompts(p)
        if prompts:
            content_items.append((p, prompts))

    if not content_items:
        raise RuntimeError("没有可用的内容图或 edit_prompts 为空")
    if not style_paths:
        raise RuntimeError("没有可用的风格图")
    if args.num_combos < len(style_paths):
        raise RuntimeError("num_combos 小于风格图数量，无法保证每种风格至少被采样一次")

    out_dir = args.out_dir.rstrip("/")
    cref_dir = join_path(out_dir, "cref")
    sref_dir = join_path(out_dir, "sref")
    smart_makedirs(cref_dir)
    smart_makedirs(sref_dir)

    rng = random.Random(args.seed)
    rng.shuffle(style_paths)
    rng.shuffle(content_items)

    prompts_map: Dict[str, str] = {}
    content_pool: List[Tuple[str, List[str]]] = []

    def refill_pool():
        content_pool.clear()
        content_pool.extend(content_items)
        rng.shuffle(content_pool)

    def next_content() -> Tuple[str, List[str]]:
        if not content_pool:
            refill_pool()
        return content_pool.pop()

    total = 0

    def write_pair(style_path: str):
        nonlocal total
        content_path, prompts = next_content()
        prompt = rng.choice(prompts)
        basename = f"{total:06d}"
        cref_out = join_path(cref_dir, basename + ".png")
        sref_out = join_path(sref_dir, basename + ".png")
        ok_c = save_png(content_path, cref_out, args.max_side)
        ok_s = save_png(style_path, sref_out, args.max_side)
        if ok_c and ok_s:
            prompts_map[basename] = prompt
            total += 1

    bar = tqdm(total=args.num_combos, unit="pair") if tqdm else None
    while total < args.num_combos:
        for sp in style_paths:
            if total >= args.num_combos:
                break
            before = total
            write_pair(sp)
            if bar and total > before:
                bar.update(total - before)
            elif (not bar) and total % 100 == 0 and total > 0:
                print(f"[PROGRESS] {total}/{args.num_combos}")

    if bar:
        bar.close()
    prompts_path = join_path(out_dir, "prompts.json")
    with mopen(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts_map, f, ensure_ascii=False, indent=2)

    print(f"[DONE] total={total} -> {out_dir}")


if __name__ == "__main__":
    main()

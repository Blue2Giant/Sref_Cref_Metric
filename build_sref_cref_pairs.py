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
/mnt/jfs/bench-bucket/sref_bench/bench_1106_content_prompt_new/
/mnt/jfs/bench-bucket/sref_bench/bench_0228_content_prompt/
python /data/benchmark_metrics/build_sref_cref_pairs.py \
  --content-dir /mnt/jfs/bench-bucket/sref_bench/bench_0228_content_prompt/ \
  --style-dir /mnt/jfs/bench-bucket/sref_bench/bench_0222_style/bench_1022_style/ \
  --out-dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_200_content  \
  --num-combos 800 \
  --max-side 2048 \
  --use-style-prompt\
  --seed 288
python /data/benchmark_metrics/build_sref_cref_pairs.py \
  --content-dir /mnt/jfs/bench-bucket/sref_bench/bench_0228_content_prompt/ \
  --style-dir /mnt/jfs/bench-bucket/sref_bench/bench_0222_style/bench_1022_style/ \
  --out-dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_sref_200_content  \
  --num-combos 800 \
  --seed 200 \
  --max-side 2048 
"""
style_list=[
  "Please apply the style of the reference image.",
  "Reference style: apply to the target image.",
  "Apply the style from the reference image.",
  "Synthesize the image using the reference style.",
  "Apply the style from the reference.",
  "Create an image in the style of the reference.",
  "Adopt the style attribute of the style reference, such as color palette and brushwork.",
  "Use the provided style reference image as a style guide.",
  "Apply the aesthetic of the style reference image.",
  "Style reference: mimic the input style.",
  "Use the style of the reference image.",
  "Use the reference image's mood and style.",
  "Render the image in the style of the reference.",
  "Follow the artistic style direction of the reference image.",
  "Make it look like the reference style.",
  "Transform the image to match the style of the reference.",
  "Apply the reference style.",
  "Use the reference image's style for generation.",
  "Apply the reference image's artistic flair.",
  "Recreate the image with the style of the provided reference.",
  "Transfer the artistic style of the reference to this image.",
  "Transfer style from the reference.",
  "Match the style of the reference image.",
  "Use the reference image to define the visual style.",
  "Generate the image using the style from the reference.",
  "Make the output follow the style of the reference.",
  "Style transfer: use the reference image's style."
]

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
        
    # Updated logic: Look for 'caption_en' which is a list of strings
    prompts = obj.get("caption_en")
    
    # Fallback to old key 'edit_prompts' if 'caption_en' is missing
    if not prompts:
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
    ap.add_argument("--use-style-prompt", action="store_true", help="If set, append a random style prompt to the content prompt.")
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
    
    # Initialize tracking for unused prompts
     # Map: content_path -> shuffled list of available prompts
    unused_prompts_map: Dict[str, List[str]] = {}
    for p, prompts in content_items:
        # Create a copy and shuffle it so the order of usage is random
        p_list = list(prompts)
        rng.shuffle(p_list)
        unused_prompts_map[p] = p_list
 
    # prompts_map: Dict[str, str] = {} # No longer needed for final dump
    content_pool: List[str] = []

    def refill_pool():
        content_pool.clear()
        # Only include paths that still have unused prompts
        valid_paths = [p for p in unused_prompts_map if unused_prompts_map[p]]
        if not valid_paths:
            raise RuntimeError("Run out of unique prompts for all content images!")
        content_pool.extend(valid_paths)
        rng.shuffle(content_pool)
 
    def next_content_pair() -> Tuple[str, str]:
        if not content_pool:
            refill_pool()
        
        # Get a path from the pool
        path = content_pool.pop()
        
        # Get the next unused prompt for this path
        prompts_list = unused_prompts_map[path]
        if not prompts_list:
            # This path is exhausted, try next one in pool
            return next_content_pair()
            
        prompt = prompts_list.pop()
        return path, prompt

    total = 0
    
    prompts_path = join_path(out_dir, "prompts.json")
    prompts_file = mopen(prompts_path, "w", encoding="utf-8")
    prompts_file.write("{\n")
    is_first_prompt = True

    def write_pair(style_path: str) -> bool:
        nonlocal total, is_first_prompt
        try:
            content_path, base_prompt = next_content_pair()
        except RuntimeError:
            return False

        # If use_style_prompt is enabled, append a random style instruction
        if args.use_style_prompt:
            style_instruction = rng.choice(style_list)
            # Ensure proper spacing/punctuation
            if not base_prompt.strip().endswith(('.', '!', '?')):
                base_prompt = base_prompt.strip() + "."
            prompt = f"{base_prompt} {style_instruction}"
        else:
            prompt = style_list[total % len(style_list)] if len(style_list) > 0 else "Apply style."

        # basename = f"{total:06d}"
        c_name = os.path.splitext(os.path.basename(content_path))[0]
        s_name = os.path.splitext(os.path.basename(style_path))[0]
        basename = f"{c_name}__{s_name}"
        
        cref_out = join_path(cref_dir, basename + ".png")
        sref_out = join_path(sref_dir, basename + ".png")
        ok_c = save_png(content_path, cref_out, args.max_side)
        ok_s = save_png(style_path, sref_out, args.max_side)
        if ok_c and ok_s:
            # Write prompt incrementally
            if not is_first_prompt:
                prompts_file.write(",\n")
            else:
                is_first_prompt = False
            
            line = f'  "{basename}": {json.dumps(prompt, ensure_ascii=False)}'
            prompts_file.write(line)
            prompts_file.flush()
            
            total += 1
        return True

    bar = tqdm(total=args.num_combos, unit="pair") if tqdm else None
    stop_early = False
    while total < args.num_combos and not stop_early:
        for sp in style_paths:
            if total >= args.num_combos:
                break
            before = total
            if not write_pair(sp):
                print(f"[WARNING] Stopping early at {total} pairs due to lack of unique prompts.")
                stop_early = True
                break
            if bar and total > before:
                bar.update(total - before)
            elif (not bar) and total % 100 == 0 and total > 0:
                print(f"[PROGRESS] {total}/{args.num_combos}")

    if bar:
        bar.close()
    
    prompts_file.write("\n}")
    prompts_file.close()

    print(f"[DONE] total={total} -> {out_dir}")


if __name__ == "__main__":
    main()

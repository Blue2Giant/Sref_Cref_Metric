#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 /data/LoraPipeline/caption_pipeline/caption_gpt4o_cref_sref.py  \
    --root /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref   --workers 16 \
    --out prompts.json 
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import time
from multiprocessing import get_context
from io import BytesIO
from PIL import Image
import requests

# ==== 使用 caption_gpt4o_demo.py 的模型配置 ====
BASE_URL = "https://models-proxy.stepfun-inc.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY", "ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig")
MODEL = "gpt-4o"
TIMEOUT = 360

# ==== 图片处理（与 caption_gpt4o_demo 同风格） ====
RESIZE_MAX_SIDE = 384
JPEG_QUALITY = 85

STYLE_TERMS = [
    r"二次元", r"动漫", r"漫画风", r"漫画", r"插画", r"插图", r"水彩", r"油画", r"丙烯", r"素描", r"速写",
    r"线稿", r"线描", r"国画", r"泼墨", r"像素风", r"赛博朋克", r"蒸汽波", r"低多边形", r"渲染",
    r"CG", r"3D", r"3D渲染", r"次世代", r"卡通", r"写实风", r"电影感", r"胶片感", r"赛璐璐", r"上色",
    r"赛璐璐上色", r"喷枪", r"版画", r"水墨", r"扁平风", r"抽象风", r"风格化", r"风格", r"画风",
    r"anime", r"manga", r"cartoon", r"illustration", r"watercolor", r"oil\s*paint(?:ing)?",
    r"sketch", r"line\s*art", r"ink", r"charcoal", r"pastel", r"pixel\s*art",
    r"low\s*poly", r"render", r"3d", r"cgi", r"stylized", r"digital\s*art",
    r"concept\s*art", r"matte\s*painting", r"cinematic", r"film\s*grain", r"bokeh",
]

# SYSTEM_PROMPT = (
#     "You are an image prompt writer. Output must be in ENGLISH ONLY.\n"
#     "Goal: craft one concise text-to-image prompt for a NEW image inspired by the given content image.\n"
#     "Rules:\n"
#     "1) Keep the same main subject(s) and key attributes from the content image, then place them in a plausible new scene.\n"
#     "2) Include action, setting, objects, and composition cues when helpful.\n"
#     "3) Do NOT use style/medium/render/camera words or mention 'reference'/'content image'.\n"
#     "4) One sentence only; specific and actionable; no second person.\n"
#     "Respond in clear, concise English only."
# )

# USER_INSTRUCTION = (
#     "Write one sentence to generate a new image inspired by the provided content image. "
#     "Keep the subject(s) and key visual attributes, and place them in a new, related situation. "
#     "Avoid style/medium/camera words and do not mention any reference."
# )
SYSTEM_PROMPT = (
    "You are a CREATIVE text-to-image prompt writer. Output must be ENGLISH ONLY.\n"
    "Goal: Based on the content of the picture, imagine a piece of text to describe a new scene. In this scene, there should be a subject related to the subject in the content picture. Please describe it in English.\n"
    "Hard rules:\n"
    "1) Do NOT describe the exact visible details from the reference; do NOT copy entities verbatim; avoid 'same/exact/identical/replicate'.\n"
    "2) Do NOT mention the reference, or 'in the image/photo/picture'.\n"
    "3) Do NOT mention art style, medium, rendering, filters, or camera terms (lens, shot, bokeh, cinematic, film grain, etc.).\n"
    "4) Output exactly ONE sentence, no lists, no extra text."
)

"""
描述的画面内容应该和内容图的差异大一点，不要和内容图完全一样。要更多样并且更加丰富。
"""
USER_INSTRUCTION = (
    "Write a single-sentence English text-to-image prompt for a new scene reimagined from the content reference. "
    "avoid style/media/camera terms."
)


STYLE_SENTENCES = [
    "Transfer the style into the style reference picture.",
    "Transfer the style into the style reference image.",
    "Transfer the style to the style reference picture.",
    "Apply the style to the style reference picture.",
    "Adopt the style from the style reference picture.",
    "Embrace the aesthetic of the style reference picture.",
    "Incorporate the style from the style reference image.",
    "Reflect the style of the style reference picture.",
    "Capture the essence of the style reference image.",
    "Utilize the style from the style reference picture.",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SESSION = None
HEADERS = None


def _guess_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "image/png"


def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_keep_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _img_to_data_uri_jpeg(img: Image.Image, quality: int = JPEG_QUALITY) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def to_data_uri_resized_local(path: str) -> str:
    img = _load_image(path)
    img = _resize_keep_long_side(img, RESIZE_MAX_SIDE)
    return _img_to_data_uri_jpeg(img, JPEG_QUALITY)


def sanitize_prompt(text: str) -> str:
    if not text:
        return ""
    clean = text.strip()
    clean = re.sub(r"^[\"'`‘’“”\s]+|[\"'`‘’“”\s]+$", "", clean)
    for term in STYLE_TERMS:
        pattern = rf"(?:{term})(?:\s*(?:风格|风))?"
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", clean)
    clean = re.sub(r"\s+$", "", clean)
    if len(clean) > 240:
        clean = clean[:240].rstrip(",.;:!? ")
    return clean or text.strip()


def build_payload(data_uri: str) -> dict:
    return {
        "model": MODEL,
        "stream": False,
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
    }


def build_prompt(core: str, idx: int) -> str:
    sentence = STYLE_SENTENCES[idx % len(STYLE_SENTENCES)]
    return f"{core}{sentence}" if core else sentence


def list_images(folder: str) -> dict:
    items = {}
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        items[base] = path
    return items


def load_existing(out_path: str) -> dict:
    if not os.path.isfile(out_path):
        return {}
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_json(out_path: str, data: dict) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_session():
    global SESSION, HEADERS
    SESSION = requests.Session()
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def worker(task):
    base, cref_path, idx = task
    t0 = time.time()
    try:
        data_uri = to_data_uri_resized_local(cref_path)
        payload = build_payload(data_uri)
        url = BASE_URL.rstrip("/") + "/chat/completions"
        resp = SESSION.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        core = sanitize_prompt(raw)
        prompt = build_prompt(core, idx)
        dt = time.time() - t0
        return base, prompt, True, dt, ""
    except Exception as e:
        dt = time.time() - t0
        return base, "", False, dt, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--cref", default="cref")
    parser.add_argument("--sref", default="sref")
    parser.add_argument("--out", default="prompts.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exists_action", choices=["skip", "overwrite", "abort"], default="skip")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = args.root
    cref_dir = os.path.join(root, args.cref)
    sref_dir = os.path.join(root, args.sref)
    out_path = os.path.join(root, args.out)

    if not os.path.isdir(cref_dir):
        raise FileNotFoundError(f"cref dir not found: {cref_dir}")
    if not os.path.isdir(sref_dir):
        raise FileNotFoundError(f"sref dir not found: {sref_dir}")

    cref_map = list_images(cref_dir)
    sref_map = list_images(sref_dir)
    common = sorted(set(cref_map.keys()) & set(sref_map.keys()))
    if args.limit and args.limit > 0:
        common = common[: args.limit]
    if not common:
        raise ValueError("No matching basenames found between cref and sref.")

    if args.dry_run:
        print(f"Matched {len(common)} image pairs.")
        return

    existing = load_existing(out_path)
    exists_action = "overwrite" if args.overwrite else args.exists_action
    if existing:
        print(f"[INFO] Found existing prompts: {len(existing)} in {out_path}")
        if exists_action == "abort":
            print("[INFO] Output already has content, abort due to exists_action=abort")
            return
        if exists_action == "skip":
            common = [b for b in common if b not in existing]
    if not common:
        print("No new items to process.")
        return

    tasks = [(base, cref_map[base], idx) for idx, base in enumerate(common)]
    ctx = get_context("spawn")
    errors = {}

    with ctx.Pool(processes=max(1, args.workers), initializer=init_session) as pool:
        for base, prompt, ok, dt, err in pool.imap_unordered(worker, tasks):
            if ok:
                existing[base] = prompt
                save_json(out_path, existing)
                print(f"[OK] {base} {dt:.2f}s")
            else:
                errors[base] = err
                print(f"[ERR] {base} {dt:.2f}s {err}")

    print(f"Saved {len(existing)} prompts to {out_path}")
    if errors:
        print(f"Failed {len(errors)} items")


if __name__ == "__main__":
    main()

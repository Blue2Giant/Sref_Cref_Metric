#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
两图内容相似度最小 demo：
- 输入两张图片路径（本地 / s3:// / oss://），评估“主体内容/主题”一致性
- 模型输出严格为一行：score@reason（score 为 0-10 整数）
"""

import os
import re
import argparse
import base64
import mimetypes
import json
from typing import Optional, Dict, Any, Tuple

from openai import OpenAI
from megfile.smart import (
    smart_open as mopen,
    smart_exists,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CONTENT_SIMILARITY_PROMPT = r"""
Rate from 0 to 10:
Evaluate how consistent Image B is with Image A in terms of SUBJECT CONTENT / THEME, regardless of style.

Important Notes:
* Scoring should be strict — avoid giving high scores unless the subject is clearly and accurately consistent.
* Ignore style differences: rendering style, brushwork, lighting mood, color grading, resolution, noise, aesthetics.
* Focus on content: who/what is present, key attributes, layout, scene category, actions, and meaningful text/logos.
* If identity/subject cannot be verified due to blur/occlusion, keep the score conservative (prefer lower).

Detailed aspects to consider:
1) Human identity & attributes:
   - Facial identity (shape, features), hairstyle/color/length, body shape/build.
   - Clothing categories, colors, patterns, logos, accessories (glasses, hats, jewelry).
2) Objects & attributes:
   - Object categories, colors, materials, sizes, and COUNT for prominent items.
   - Presence/absence of key props; text/logo content where legible.
3) Spatial layout & composition:
   - Relative positions of major elements; subject scale and viewpoint within reasonable tolerance.
4) Background / scene category:
   - Indoor/outdoor; place type; major structures/furniture/landmarks.
5) Actions / poses / interactions:
   - Human/animal/body poses; gesture semantics.
6) Text / logos / symbols:
   - Words/numbers/symbols important to scene meaning.

Score rubric (0–10):
* 0: Completely unrelated.
* 1–3: Mostly inconsistent; minimal overlap.
* 4–6: Partially consistent with significant mismatches.
* 7–9: Mostly consistent with minor issues.
* 10: Fully consistent; all major content aligns.

Output rules (very important):
* Output ONLY one line in the format: score@reason
* score must be an integer from 0 to 10.
* reason must be 1-2 short sentences, specific and observable.
""".strip()

def path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with mopen(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    m = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, flags=re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    return s

def _validate_image_path(p: str) -> str:
    if p.startswith(("s3://", "oss://")):
        if not smart_exists(p):
            raise FileNotFoundError(f"找不到文件: {p}")
        ext = os.path.splitext(p)[1].lower()
        if ext and (ext not in IMG_EXTS):
            pass
        return p

    if not os.path.isfile(p):
        raise FileNotFoundError(f"找不到文件: {p}")
    ext = os.path.splitext(p)[1].lower()
    if ext and (ext not in IMG_EXTS):
        pass
    return p

def build_messages(img_a: str, img_b: str):
    content = []
    content.append({"type": "text", "text": "Image A:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_a)}})
    content.append({"type": "text", "text": "Image B:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_b)}})
    content.append({"type": "text", "text": CONTENT_SIMILARITY_PROMPT})
    return [{"role": "user", "content": content}]

def _clamp_score_0_10(score: int) -> int:
    if score < 0:
        return 0
    if score > 10:
        return 10
    return score

def parse_score_reason(raw_text: str) -> Dict[str, Any]:
    clean = strip_code_fences(raw_text).strip()
    if not clean:
        raise ValueError("模型输出为空")
    first_line = clean.splitlines()[0].strip()
    m = re.match(r"^\s*(\d{1,2})\s*@\s*(.+?)\s*$", first_line)
    if not m:
        raise ValueError(f"输出不符合 score@reason: {first_line!r}")
    score = _clamp_score_0_10(int(m.group(1)))
    reason = m.group(2).strip()
    if not reason:
        raise ValueError("reason 为空")
    return {"score": score, "reason": reason}

def run_content_score(
    client: OpenAI,
    model: str,
    img_a: str,
    img_b: str,
    max_tokens: int = 128,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    messages = build_messages(img_a, img_b)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = parse_score_reason(raw)
    except Exception as e:
        return raw, None, {"error": str(e)}
    return raw, parsed, None

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 两图内容相似度：score@reason")
    parser.add_argument("--img-a", required=True, help="图片A路径（本地 / s3:// / oss://）")
    parser.add_argument("--img-b", required=True, help="图片B路径（本地 / s3:// / oss://）")
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI 兼容 base_url，如 http://host:port/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--model",
        default="Qwen3-VL-30B-A3B-Instruct",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
    )
    parser.add_argument("--print-debug", action="store_true")

    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    img_a = _validate_image_path(args.img_a)
    img_b = _validate_image_path(args.img_b)

    raw, parsed, err = run_content_score(
        client=client,
        model=args.model,
        img_a=img_a,
        img_b=img_b,
        max_tokens=args.max_tokens,
    )

    print("=== Raw output ===")
    print(raw)

    if parsed is None:
        print("\n[ERROR] 无法解析模型输出为 score@reason。")
        if args.print_debug:
            print(json.dumps(err, ensure_ascii=False, indent=2))
        return

    print("\n=== Parsed ===")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

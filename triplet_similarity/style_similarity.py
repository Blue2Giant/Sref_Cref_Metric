#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
两图画风相似性：一次请求解析总体 score 与 reason（不需要 logits）

用法示例：
python style_similarity.py \
  --img-a a.png \
  --img-b b.png \
  --weights 0.25 0.2 0.25 0.15 0.15 \
  --base-url http://10.191.0.41:22002/v1 \
  --model Qwen3-VL-30B-A3B-Instruct \
  --print-debug

weights 顺序固定为：
[brushstroke, texture, color, shape, pattern]
"""

import os
import re
import json
import argparse
import base64
import mimetypes
from typing import Optional, Dict, Any, Tuple

from openai import OpenAI

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ========= Prompt（更清晰的定义 + 严格输出格式） =========
STYLE_5SCORE_PROMPT = r"""
你将看到两张图片：A 和 B。请只从“画风/风格”角度评估它们有多一致。
不要根据主体内容是否相同来判断（允许人物/物体/场景完全不同）。

你需要综合所有纬度的基础上总体的相似度给出 0 到 10 的评分。

维度定义（请严格按以下含义判断）：
1) BRUSHSTROKE（笔触/线条/边缘处理）
   - 线条粗细、线条是否抖动、勾边方式、笔刷痕迹是否明显
   - 边缘是硬边/软边、是否有描边、涂抹/泼洒/铅笔/钢笔等笔触特征
2) TEXTURE（纹理/材质表现/颗粒与噪声/画布纸张质感）
   - 表面微观颗粒感：例如“砂纸颗粒/胶片噪声/水彩纸纤维/油画布纹”
   - 材质的微细结构呈现方式：例如“涂料堆叠纹/喷点/网点/浮雕质感”
   - 注意：TEXTURE 不是配色，也不是形状轮廓
3) COLOR（配色体系与色彩分布/色温/饱和度/对比度）
   - 主色调与色温倾向（偏冷/偏暖）、饱和度高低、明暗对比强弱
   - 色彩分布相似：比如“主要颜色的组成比例/覆盖面积是否相近（大面积背景色、主色块比例）”
   - 注意：COLOR 不是纹理颗粒，也不是笔触线条
4) SHAPE（形状语言/造型习惯）
   - 几何化/写实化程度、比例是否夸张、轮廓更锐利还是更圆润
   - 结构简化方式（例如卡通扁平、Q版、极简几何、写实结构）
5) PATTERN（模式/母题/装饰性图案化规律）
   - 是否有重复纹样、装饰元素、固定符号化母题（如固定花纹、反复出现的装饰线、图案化背景）
   - 图案密度、重复规律、装饰性元素的组织方式是否相似

评分标准（0-10，评分要严格）：
* 0：风格完全不一致，基本无法认为在该维度相同。
* 1–3：大部分不一致，只有很弱/偶然的相似。
* 4–6：有部分相似，但存在明显缺失或关键差异。
* 7–9：大体一致，仅有少量小问题。
* 10：完全一致，关键特征高度匹配。

重要说明（必须遵守）：
* 评分要严格，除非该维度清晰且准确匹配，否则不要给高分。
* 只评估“风格维度”是否一致；不要考虑人物身份是否相同、物体是否同一个、背景/场景是否一致。
* 不要评估审美好坏或画面好看与否。

输出规则（非常重要）：
* score 必须是 0-10 的整数。
* reason 1-2 句，指出可观察到的具体依据。
* 严格输出格式为 score@reason（只输出这一行，不要输出其它任何字符/标点/换行）

""".strip()

# ----------------- utils -----------------
def path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_messages(img_a: str, img_b: str):
    content = []
    content.append({"type": "text", "text": "Image A:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_a)}})
    content.append({"type": "text", "text": "Image B:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_b)}})
    content.append({"type": "text", "text": STYLE_5SCORE_PROMPT})
    return [{"role": "user", "content": content}]

def _validate_image_path(p: str) -> str:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"找不到文件: {p}")
    ext = os.path.splitext(p)[1].lower()
    if ext and (ext not in IMG_EXTS):
        pass
    return p

def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    m = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, flags=re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    return s

def _clamp_score_0_10(score: int) -> int:
    if score < 0:
        return 0
    if score > 10:
        return 10
    return score

def parse_overall_score_reason(raw_text: str) -> Dict[str, Any]:
    clean = strip_code_fences(raw_text).strip()
    if not clean:
        raise ValueError("模型输出为空")

    first_line = clean.splitlines()[0].strip()
    m = re.match(r"^\s*(\d{1,2})\s*@\s*(.+?)\s*$", first_line)
    if m:
        score = _clamp_score_0_10(int(m.group(1)))
        reason = m.group(2).strip()
        if not reason:
            raise ValueError("reason 为空")
        return {"score": score, "reason": reason}

    try:
        obj = json.loads(clean)
    except Exception:
        raise ValueError(f"无法解析 score@reason，且不是 JSON：{first_line!r}")

    if not isinstance(obj, dict):
        raise ValueError("模型 JSON 输出不是 object")
    overall = obj.get("OVERALL", None)
    if not isinstance(overall, dict):
        raise ValueError("模型 JSON 缺少 OVERALL object")
    score_raw = overall.get("score", None)
    reason_raw = overall.get("reason", "")
    if isinstance(score_raw, bool) or score_raw is None:
        raise ValueError("OVERALL.score 无效")
    score = int(float(score_raw)) if isinstance(score_raw, str) else int(score_raw)
    score = _clamp_score_0_10(score)
    reason = reason_raw if isinstance(reason_raw, str) else str(reason_raw)
    reason = reason.strip()
    if not reason:
        raise ValueError("OVERALL.reason 为空")
    return {"score": score, "reason": reason}

# ----------------- core -----------------
def run_style_score_onecall(
    client: OpenAI,
    model: str,
    img_a: str,
    img_b: str,
    max_tokens: int = 512,
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
        overall = parse_overall_score_reason(raw)
    except Exception as e:
        return raw, None, {"error": str(e)}

    return raw, overall, None

# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 两图画风相似性：总体 score@reason")
    parser.add_argument("--img-a", required=True, help="图片A路径")
    parser.add_argument("--img-b", required=True, help="图片B路径")

    parser.add_argument("--base-url", required=True, help="OpenAI 兼容 base_url，如 http://host:port/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"), help="API Key")
    parser.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--timeout", type=int, default=600)

    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--print-debug", action="store_true")

    args = parser.parse_args()

    img_a = _validate_image_path(args.img_a)
    img_b = _validate_image_path(args.img_b)

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    raw, overall, err = run_style_score_onecall(
        client=client,
        model=args.model,
        img_a=img_a,
        img_b=img_b,
        max_tokens=args.max_tokens,
    )

    print("=== Raw output ===")
    print(raw)

    if overall is None:
        print("\n[ERROR] 无法解析模型输出为 score@reason。")
        if args.print_debug:
            print(json.dumps(err, ensure_ascii=False, indent=2))
        return

    print("\n=== Overall ===")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    if args.print_debug:
        print("\n=== Debug ===")
        print(json.dumps({"parsed": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

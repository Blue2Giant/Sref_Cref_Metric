#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import base64
import random
import argparse
from io import BytesIO
from typing import Optional, Tuple, Dict, List

import requests
from PIL import Image
"""
python /data/LoraPipeline/sref_pipeline/qwen3_style_batch_check_rs.py  --input_txt /mnt/jfs/xhs_style_dir/image_paths.txt  --output_dir /mnt/jfs/style_check_results_rs  --num_samples 30000
"""
API_KEY = "EMPTY"
MODEL = "Qwen3-VL-30B-A3B-Instruct"
BASE_URL = "http://10.191.6.43:22002/v1"

#xp
# MODEL = "Qwen3VL30BA3B-Image-Edit"
# BASE_URL = "http://stepcast-router.shai-core:9200/v1"

TIMEOUT = 180

RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85

NUM_SAMPLE_A = 30000
FAR_DISTANCE_MIN_OFFSET = 6

ANALYSIS_SAMPLES = 3
ANALYSIS_TEMPERATURE = 0.7

JUDGE_SAMPLES = 5
JUDGE_TEMPERATURE = 0.7
JUDGE_CONF_THRESHOLD = 0.8

MIN_DESC_TOTAL_LEN = 100

Image.MAX_IMAGE_PIXELS = None

ANALYSIS_SYSTEM_PROMPT = (
    "你是一个专业的视觉艺术与设计风格分析专家。\n"
    "你的任务是对输入的图片进行深度风格拆解，输出详细的分析描述。\n"
    "不要对画面语义内容进行描述，只关注视觉风格。\n"
    "请不要输出任何多余的寒暄或解释，只输出 JSON。"
)

ANALYSIS_USER_PROMPT = (
    "请对这张图片进行详细的视觉风格分析。请从以下几个维度进行描述：\n\n"
    "1. 纹理与材质：画面的质感（如粗糙、细腻、油画感、水彩感、数码绘图感、胶片颗粒感等）。\n"
    "2. 色彩运用：整体色调（冷暖、饱和度、对比度）、特定的配色方案或色彩倾向。\n"
    "3. 笔触与线条：线条的粗细、锐利度、笔触的可见性、描边风格（如无描边、粗黑边等）。\n"
    "4. 光影处理：光照来源、阴影强度、立体感渲染方式（如二次元平涂、厚涂、真实光影等）。\n"
    "5. 几何构造与形态：主体的造型特征（如写实、夸张、Q版/Chibi、极简、抽象等）。如果是Q版，请描述头身比、面部特征。\n"
    "6. 构图与视角：画面的布局方式、透视感。\n\n"
    "请严格只返回一个 JSON 对象，格式如下：\n"
    "{\n"
    "  \"texture\": \"纹理与材质的描述...\",\n"
    "  \"color\": \"色彩运用的描述...\",\n"
    "  \"lines\": \"笔触与线条的描述...\",\n"
    "  \"lighting\": \"光影处理的描述...\",\n"
    "  \"geometry\": \"几何构造与形态的描述...\",\n"
    "  \"composition\": \"构图与视角的描述...\"\n"
    "}\n"
)

JUDGE_SYSTEM_PROMPT = (
    "你是一个严格的视觉风格一致性评估裁判。\n"
    "你的任务是根据两段风格描述，判断两张图片的风格是否高度一致。\n"
    "只输出 JSON。"
)

JUDGE_USER_PROMPT_TEMPLATE = """
以下是两张图片的风格分析数据（JSON格式）：

【图片 A 风格数据】：
{desc_a}

【图片 B 风格数据】：
{desc_b}

请逐项对比这两组数据的各个维度（纹理、色彩、线条、光影、几何、构图），判断图片 B 的风格是否与图片 A **高度一致**。

判定标准：
1. 核心画风必须相同：例如图片 A 是“Q版二次元平涂”，图片 B 也必须是；如果是“写实厚涂”，图片 B 也必须是。
2. 关键视觉特征必须匹配：
   - 线条：粗细、描边风格必须一致。
   - 光影：上色方式（平涂vs厚涂）、立体感渲染逻辑必须一致。
   - 纹理：画面质感（如水彩、油画、数码）必须一致。
3. 允许内容不同：主体人物、动作、背景可以不同，但“渲染风格”和“视觉质感”必须一致。
4. 容忍度：对于微小的色调差异或构图差异可以容忍，但对于画风类别（如Q版变写实）的差异必须判为 False。

请严格只返回一个 JSON 对象，格式如下：
{{
  "is_consistent": true,
  "reason": "简短的判断理由，指出具体哪个维度不一致或均一致"
}}
"""


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _load_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        log(f"[Warn] 无法读取图片 {path}: {e}")
        return None


def _resize_keep_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def get_image_data_uri(path: str) -> Optional[str]:
    img = _load_image(path)
    if img is None:
        return None

    img = _resize_keep_long_side(img, RESIZE_MAX_SIDE)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return s


def concat_images(path_a: str, path_b: str) -> Optional[Image.Image]:
    img_a = _load_image(path_a)
    img_b = _load_image(path_b)
    if not img_a or not img_b:
        return None

    h = max(img_a.height, img_b.height)

    if img_a.height != h:
        scale = h / img_a.height
        img_a = img_a.resize((int(img_a.width * scale), h), Image.LANCZOS)

    if img_b.height != h:
        scale = h / img_b.height
        img_b = img_b.resize((int(img_b.width * scale), h), Image.LANCZOS)

    new_w = img_a.width + img_b.width
    new_img = Image.new("RGB", (new_w, h))
    new_img.paste(img_a, (0, 0))
    new_img.paste(img_b, (img_a.width, 0))

    return new_img


def call_qwen_chat(messages: list, temperature: float, max_retries: int = 2, retry_delay: float = 2.0) -> Optional[str]:
    payload = {
        "model": MODEL,
        "temperature": float(temperature),
        "messages": messages,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                BASE_URL.rstrip("/") + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            resp_json = resp.json()
            content = resp_json["choices"][0]["message"]["content"]

            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif isinstance(c, str):
                        parts.append(c)
                content = "\n".join(parts)

            return content

        except Exception as e:
            log(f"[Err] API 请求出错(第 {attempt + 1} 次): {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return None


def validate_style_json(obj: Dict) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = ["texture", "color", "lines", "lighting", "geometry", "composition"]
    for k in keys:
        if k not in obj:
            return False
        v = obj.get(k)
        if not isinstance(v, str):
            return False
    total_len = sum(len(obj[k]) for k in keys)
    if total_len < MIN_DESC_TOTAL_LEN:
        return False
    return True


def style_total_length(obj: Dict) -> int:
    keys = ["texture", "color", "lines", "lighting", "geometry", "composition"]
    return sum(len(str(obj.get(k, ""))) for k in keys)


def analyze_image_style_single(img_path: str) -> Optional[Dict[str, str]]:
    log(f"正在分析风格: {os.path.basename(img_path)} ...")
    data_uri = get_image_data_uri(img_path)
    if not data_uri:
        return None

    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ANALYSIS_USER_PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]

    content = call_qwen_chat(messages, temperature=ANALYSIS_TEMPERATURE)
    if not content:
        return None

    try:
        clean = strip_code_fences(content)
        obj = json.loads(clean)
        if validate_style_json(obj):
            return obj
        return None
    except json.JSONDecodeError:
        log(f"[Warn] Analyze JSON 解析失败: {content[:100]}...")
        return None


def analyze_image_style(img_path: str) -> Optional[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    for _ in range(ANALYSIS_SAMPLES):
        obj = analyze_image_style_single(img_path)
        if obj:
            candidates.append(obj)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    best = max(candidates, key=style_total_length)
    return best


def judge_consistency(desc_a: Dict, desc_b: Dict) -> Tuple[Optional[bool], str]:
    str_a = json.dumps(desc_a, ensure_ascii=False, indent=2) if isinstance(desc_a, dict) else str(desc_a)
    str_b = json.dumps(desc_b, ensure_ascii=False, indent=2) if isinstance(desc_b, dict) else str(desc_b)

    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        desc_a=str_a,
        desc_b=str_b
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    votes_true = 0
    votes_false = 0
    reasons_true: List[str] = []
    reasons_false: List[str] = []

    for i in range(JUDGE_SAMPLES):
        content = call_qwen_chat(messages, temperature=JUDGE_TEMPERATURE)
        if not content:
            continue
        try:
            clean = strip_code_fences(content)
            obj = json.loads(clean)
            is_cons = bool(obj.get("is_consistent", False))
            reason = obj.get("reason", "")
            if is_cons:
                votes_true += 1
                reasons_true.append(reason)
            else:
                votes_false += 1
                reasons_false.append(reason)
        except json.JSONDecodeError:
            log(f"[Warn] Judge JSON 解析失败: {content[:100]}...")
            continue

    total_votes = votes_true + votes_false
    if total_votes == 0:
        return None, "无有效判别结果"

    if votes_true >= votes_false:
        majority_label = True
        majority_votes = votes_true
        majority_reasons = reasons_true
        minority_votes = votes_false
    else:
        majority_label = False
        majority_votes = votes_false
        majority_reasons = reasons_false
        minority_votes = votes_true

    confidence = majority_votes / float(total_votes)

    if confidence < JUDGE_CONF_THRESHOLD:
        reason = f"无法判断：低置信度投票 true={votes_true}, false={votes_false}, confidence={confidence:.2f}"
        return None, reason

    chosen_reason = majority_reasons[0] if majority_reasons else ""
    reason = (
        f"多裁判一致：is_consistent={majority_label}, "
        f"confidence={confidence:.2f}, 示例理由：{chosen_reason}"
    )
    return majority_label, reason


def main():
    parser = argparse.ArgumentParser(description="批量风格一致性检查（带拒绝采样）")
    parser.add_argument("--input_txt", required=True, help="包含图片路径的txt文件")
    parser.add_argument("--output_dir", required=True, help="结果输出目录")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLE_A, help="随机抽样 A 图的数量")
    args = parser.parse_args()

    input_txt = args.input_txt
    output_dir = args.output_dir
    num_samples = args.num_samples

    if not os.path.exists(input_txt):
        log(f"[Err] 输入文件不存在: {input_txt}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log(f"[Info] 创建输出目录: {output_dir}")

    image_paths: List[str] = []
    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                image_paths.append(line)

    total_imgs = len(image_paths)
    if total_imgs == 0:
        log("[Info] 没有找到图片路径")
        sys.exit(0)

    log(f"[Info] 共读取到 {total_imgs} 张图片")

    actual_samples = min(num_samples, total_imgs)
    sampled_indices = sorted(random.sample(range(total_imgs), actual_samples))
    log(f"[Info] 随机抽样 {actual_samples} 张图片作为 Image A")

    style_cache: Dict[str, Dict] = {}

    def get_style_desc(path: str) -> Optional[Dict]:
        if path in style_cache:
            return style_cache[path]
        desc = analyze_image_style(path)
        if desc:
            style_cache[path] = desc
        return desc

    count = 0

    for idx_a in sampled_indices:
        path_a = image_paths[idx_a]

        candidate_indices: List[int] = []

        if idx_a - 1 >= 0:
            candidate_indices.append(idx_a - 1)

        if idx_a + 1 < total_imgs:
            candidate_indices.append(idx_a + 1)

        far_candidates: List[int] = []
        if idx_a - FAR_DISTANCE_MIN_OFFSET >= 0:
            far_candidates.extend(range(0, idx_a - FAR_DISTANCE_MIN_OFFSET + 1))
        if idx_a + FAR_DISTANCE_MIN_OFFSET < total_imgs:
            far_candidates.extend(range(idx_a + FAR_DISTANCE_MIN_OFFSET, total_imgs))

        if far_candidates:
            far_idx = random.choice(far_candidates)
            candidate_indices.append(far_idx)

        if not candidate_indices:
            log(f"[Skip] 图片 A (index={idx_a}) 周围没有足够的图片进行对比")
            continue

        desc_a = get_style_desc(path_a)
        if not desc_a:
            log(f"[Skip] 无法分析图 A: {path_a}")
            continue

        for idx_b in candidate_indices:
            path_b = image_paths[idx_b]

            desc_b = get_style_desc(path_b)
            if not desc_b:
                log(f"[Skip] 无法分析图 B: {path_b}")
                continue

            log(f"正在对比: A[{idx_a}] {os.path.basename(path_a)} vs B[{idx_b}] {os.path.basename(path_b)}")
            is_consistent, reason = judge_consistency(desc_a, desc_b)

            base_name = f"A{idx_a:04d}_B{idx_b:04d}_{os.path.splitext(os.path.basename(path_a))[0]}_VS_{os.path.splitext(os.path.basename(path_b))[0]}"

            result_data = {
                "image_a": {
                    "index": idx_a,
                    "path": path_a
                },
                "image_b": {
                    "index": idx_b,
                    "path": path_b
                },
                "style_desc_a": desc_a,
                "style_desc_b": desc_b,
                "is_consistent": is_consistent,
                "reason": reason
            }
            json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            concat_img = concat_images(path_a, path_b)
            if concat_img:
                img_save_path = os.path.join(output_dir, f"{base_name}.jpg")
                concat_img.save(img_save_path, quality=85)

            log(f"   -> 结果已保存: {is_consistent} | {reason}")
            count += 1

    log(f"\n[Done] 全部完成，共生成 {count} 组对比结果。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
import json
import time
import base64
import argparse
import math
from io import BytesIO
from typing import Dict, Optional, List, Tuple, Any

import requests
from PIL import Image, ImageDraw, ImageFont

API_KEY = "EMPTY"
MODEL = "Qwen3VL30BA3B-Image-Edit"
BASE_URL = "http://stepcast-router.shai-core:9200/v1"
TIMEOUT = 180

RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85

# 置信度阈值：预测类别的 softmax 概率 > 阈值才接受，否则 reject
JUDGE_CONF_THRESHOLD = 0.8

Image.MAX_IMAGE_PIXELS = None

SYSTEM_PROMPT_01 = (
    "你是一个只关注“画风和视觉风格”的严格评审。\n"
    "你不会考虑画面里的人物身份、动作含义、故事语义，只看画风本身。\n"
    "你的任务：判断两张图片在画风上是否高度一致。\n"
    "你必须只输出一个字符：0 或 1。\n"
    "1 表示画风高度一致；0 表示画风不一致。\n"
    "不要输出任何多余文字、空格、换行、JSON。"
)

USER_INSTRUCTION_01 = (
    "请仅从“画风 / 视觉风格”的角度比较图片 A 和图片 B，忽略画面内容语义。\n\n"
    "判断维度：纹理与材质、色彩运用、笔触与线条、光影与体积、几何造型风格、构图与视角。\n"
    "如果大部分关键维度（尤其几何风格+光影+线条/材质）一致，则输出 1；否则输出 0。\n"
    "最终只输出一个字符：0 或 1。"
)


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


def draw_mark(img: Image.Image, human_label, model_label) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    short_side = min(w, h)
    font_size = max(16, int(short_side * 0.2))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if isinstance(model_label, bool) and human_label is not None:
        agree = (human_label == model_label)
        if agree:
            text = "✔"
            color = (0, 200, 0)
        else:
            text = "✘"
            color = (220, 0, 0)
    else:
        text = None
        color = (255, 215, 0)

    margin = max(10, int(short_side * 0.05))
    if text:
        position = (margin, margin)
        draw.text(position, text, fill=color, font=font)
    else:
        radius = max(8, int(short_side * 0.1))
        cx = margin + radius
        cy = margin + radius
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, outline=color, width=max(3, int(radius * 0.15)))
    return out


def build_image_map(txt_path: str) -> Dict[str, str]:
    image_map: Dict[str, str] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            base = os.path.splitext(os.path.basename(p))[0]
            if base not in image_map:
                image_map[base] = p
    return image_map


def parse_json_filename(fname: str) -> Optional[Dict[str, str]]:
    name = os.path.splitext(fname)[0]
    if "_VS_" not in name:
        return None
    left, base_b = name.split("_VS_", 1)
    parts = left.split("_", 2)
    if len(parts) < 3:
        return None
    base_a = parts[2]
    return {"base_a": base_a, "base_b": base_b, "stem": name}


def _safe_exp(x: float) -> float:
    # 防溢出
    if x > 60:
        return math.exp(60)
    if x < -60:
        return math.exp(-60)
    return math.exp(x)


def _softmax2(logp0: float, logp1: float) -> Tuple[float, float]:
    # 返回 p0, p1
    m = max(logp0, logp1)
    a0 = _safe_exp(logp0 - m)
    a1 = _safe_exp(logp1 - m)
    denom = a0 + a1
    if denom <= 0:
        return 0.5, 0.5
    return a0 / denom, a1 / denom


def call_qwen_chat_raw(
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 1,
    need_logprobs: bool = True,
    top_logprobs: int = 8,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "model": MODEL,
        "temperature": float(temperature),
        "messages": messages,
        "max_tokens": int(max_tokens),
    }

    # 尽量请求 logprobs + top_logprobs（不同后端字段名可能不同，这里都带上）
    if need_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(top_logprobs)
        # 某些兼容实现会用这个名字
        payload["top_k"] = int(top_logprobs)

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
            return resp.json()
        except Exception as e:
            log(f"[Err] API 请求出错(第 {attempt + 1} 次): {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return None


def _extract_text_from_choice(choice: Dict[str, Any]) -> str:
    # 兼容 message.content 可能是 str 或 list (多模态)
    msg = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    return str(content)


def _collect_top_logprobs_mapping(resp_json: Dict[str, Any]) -> Dict[str, float]:
    """
    尽量从不同实现的返回里提取“第一个生成 token”的 top_logprobs 映射：token->logprob
    兼容：
      - OpenAI 新式：choices[0].logprobs.content[0].top_logprobs = [{token, logprob}, ...]
      - OpenAI/兼容：choices[0].logprobs = {tokens, token_logprobs, top_logprobs}
      - 某些实现：choice["message"]["logprobs"] ...
    """
    mapping: Dict[str, float] = {}

    if not resp_json:
        return mapping
    choices = resp_json.get("choices", [])
    if not choices:
        return mapping
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    logprobs = choice0.get("logprobs", None)

    # 有些实现把 logprobs 放进 message 里
    if logprobs is None:
        msg = choice0.get("message", {}) if isinstance(choice0.get("message"), dict) else {}
        logprobs = msg.get("logprobs", None)

    if not isinstance(logprobs, dict):
        return mapping

    # 1) OpenAI 新式：logprobs["content"][0]["top_logprobs"]
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        first = content[0] if isinstance(content[0], dict) else None
        if isinstance(first, dict):
            top = first.get("top_logprobs")
            if isinstance(top, list):
                for item in top:
                    if isinstance(item, dict):
                        tok = item.get("token")
                        lp = item.get("logprob")
                        if isinstance(tok, str) and isinstance(lp, (int, float)):
                            mapping[tok] = float(lp)
                if mapping:
                    return mapping
            if isinstance(top, dict):
                # 少见：直接 token->logprob
                for tok, lp in top.items():
                    if isinstance(tok, str) and isinstance(lp, (int, float)):
                        mapping[tok] = float(lp)
                if mapping:
                    return mapping

    # 2) vLLM/旧式：logprobs["top_logprobs"] 可能是 list[dict]，取第 0 个 token 的 dict
    top_lp = logprobs.get("top_logprobs")
    if isinstance(top_lp, list) and top_lp:
        first = top_lp[0]
        if isinstance(first, dict):
            for tok, lp in first.items():
                if isinstance(tok, str) and isinstance(lp, (int, float)):
                    mapping[tok] = float(lp)
            if mapping:
                return mapping
        if isinstance(first, list):
            # list[{token, logprob}] 形式
            for item in first:
                if isinstance(item, dict):
                    tok = item.get("token")
                    lp = item.get("logprob")
                    if isinstance(tok, str) and isinstance(lp, (int, float)):
                        mapping[tok] = float(lp)
            if mapping:
                return mapping

    # 3) 兜底：如果只有 tokens/token_logprobs，没有 top 候选，则无法得到 0/1 两者
    return mapping


def _extract_01_logprobs(resp_json: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    """
    返回 logp0, logp1（如果能拿到），以及原始 mapping（便于 debug）
    注意 token 可能是 "0" / "1" / " 0" / " 1" 等，需要 strip 后匹配。
    """
    mapping_raw = _collect_top_logprobs_mapping(resp_json)
    logp0 = None
    logp1 = None

    for tok, lp in mapping_raw.items():
        t = tok.strip()
        if t == "0":
            logp0 = lp
        elif t == "1":
            logp1 = lp

    return logp0, logp1, mapping_raw


def direct_judge_images_01(path_a: str, path_b: str) -> Tuple[Optional[bool], str, Optional[float]]:
    """
    返回：
      - label: True/False 表示一致/不一致；None 表示拒绝采样
      - reason: 简短原因/拒绝原因
      - confidence: 预测类别的 softmax 概率（0~1），拿不到则 None
    """
    data_a = get_image_data_uri(path_a)
    data_b = get_image_data_uri(path_b)
    if not data_a or not data_b:
        return None, "图片编码失败", None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_01},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_INSTRUCTION_01},
                {"type": "text", "text": "下面是图片 A："},
                {"type": "image_url", "image_url": {"url": data_a}},
                {"type": "text", "text": "下面是图片 B："},
                {"type": "image_url", "image_url": {"url": data_b}},
                {"type": "text", "text": "只输出一个字符：0 或 1。"},
            ],
        },
    ]

    resp_json = call_qwen_chat_raw(
        messages,
        temperature=0.0,
        max_tokens=1,
        need_logprobs=True,
        top_logprobs=8,
    )
    if not resp_json:
        return None, "API 无响应/请求失败 -> reject", None

    choices = resp_json.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return None, "返回结构异常(无 choices) -> reject", None

    text = strip_code_fences(_extract_text_from_choice(choices[0])).strip()
    # max_tokens=1 仍可能出现诸如 "1\n" 或带空格的 token，这里取首个非空白字符
    pred_char = None
    for ch in text:
        if not ch.isspace():
            pred_char = ch
            break

    if pred_char not in ("0", "1"):
        return None, f"输出不是 0/1 (got={text!r}) -> reject", None

    pred_is_consistent = (pred_char == "1")

    logp0, logp1, mapping_raw = _extract_01_logprobs(resp_json)
    if logp0 is None or logp1 is None:
        # 拿不到 0/1 两者的 logprob 就无法算置信度，按你的要求直接 reject
        return None, f"无法提取 0/1 top_logprobs -> reject (keys={list(mapping_raw.keys())[:8]})", None

    p0, p1 = _softmax2(logp0, logp1)
    conf = p1 if pred_is_consistent else p0

    if conf <= JUDGE_CONF_THRESHOLD:
        return None, f"低置信度 conf={conf:.3f} (p0={p0:.3f}, p1={p1:.3f}) -> reject", conf

    reason = f"接受：pred={pred_char}, conf={conf:.3f} (p0={p0:.3f}, p1={p1:.3f})"
    return pred_is_consistent, reason, conf


def main():
    global MODEL, BASE_URL, JUDGE_CONF_THRESHOLD
    parser = argparse.ArgumentParser(description="用 0/1 单 token 判别风格一致性，并用 0/1 token logprob 计算置信度")
    parser.add_argument("--image_txt", required=True, help="包含图片完整路径的txt文件")
    parser.add_argument("--json_dir", required=True, help="原始结果JSON所在目录")
    parser.add_argument("--output_dir", required=True, help="新的结果输出目录")
    parser.add_argument("--model", type=str, default=MODEL, help="Qwen 模型名称")
    parser.add_argument("--base_url", type=str, default=BASE_URL, help="Qwen 接口 base url")
    parser.add_argument("--conf_threshold", type=float, default=JUDGE_CONF_THRESHOLD, help="置信度阈值(>该值才接受)")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在的输出文件")
    args = parser.parse_args()

    MODEL = args.model
    BASE_URL = args.base_url

    JUDGE_CONF_THRESHOLD = float(args.conf_threshold)

    image_txt = args.image_txt
    json_dir = args.json_dir
    output_dir = args.output_dir
    overwrite = args.overwrite

    if not os.path.exists(image_txt):
        log(f"[Err] image_txt 不存在: {image_txt}")
        sys.exit(1)
    if not os.path.isdir(json_dir):
        log(f"[Err] json_dir 非目录或不存在: {json_dir}")
        sys.exit(1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log(f"[Info] 创建输出目录: {output_dir}")

    log("[Info] 构建图片路径索引")
    image_map = build_image_map(image_txt)
    log(f"[Info] 索引中共有 {len(image_map)} 个唯一图片名")

    json_files: List[str] = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]
    json_files.sort()
    total_json = len(json_files)
    log(f"[Info] 在 json_dir 中发现 {total_json} 个 json 文件")

    agree_true = 0
    agree_false = 0
    reject_cnt = 0

    for idx, fname in enumerate(json_files):
        info = parse_json_filename(fname)
        if not info:
            log(f"[Skip] 文件名无法解析为图片对: {fname}")
            continue
        json_path = os.path.join(json_dir, fname)

        base_a = info["base_a"]
        base_b = info["base_b"]
        stem = info["stem"]

        path_a = image_map.get(base_a)
        path_b = image_map.get(base_b)
        if not path_a or not path_b:
            log(f"[Skip] 无法在 image_txt 中找到图片路径: {base_a} 或 {base_b}")
            continue

        out_json_path = os.path.join(output_dir, fname)
        out_img_path = os.path.join(output_dir, stem + ".jpg")
        if not overwrite and os.path.exists(out_json_path) and os.path.exists(out_img_path):
            log(f"[Skip] 输出已存在且未开启覆盖: {fname}")
            continue

        log(f"[{idx+1}/{total_json}] 处理: {fname}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                original = json.load(f)
        except Exception as e:
            log(f"[Skip] 读取 JSON 失败 {fname}: {e}")
            continue

        human_label = original.get("human", None)

        model_is_consistent, model_reason, model_conf = direct_judge_images_01(path_a, path_b)

        agree = None
        if isinstance(model_is_consistent, bool) and human_label is not None:
            agree = (human_label == model_is_consistent)
            if agree:
                agree_true += 1
            else:
                agree_false += 1
        else:
            reject_cnt += 1

        new_data = dict(original)
        new_data["human"] = human_label
        new_data["image_a_path"] = path_a
        new_data["image_b_path"] = path_b

        if isinstance(model_is_consistent, bool):
            new_data["is_consistent_new"] = model_is_consistent
        else:
            new_data["is_consistent_new"] = "reject"   # 拒绝采样
        new_data["reason_new"] = model_reason
        new_data["confidence"] = model_conf
        new_data["agree"] = agree

        try:
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"[Warn] 写入 JSON 失败 {out_json_path}: {e}")

        concat_path = os.path.join(json_dir, stem + ".jpg")
        concat_img: Optional[Image.Image] = None
        if os.path.exists(concat_path):
            try:
                concat_img = Image.open(concat_path).convert("RGB")
            except Exception as e:
                log(f"[Warn] 读取原拼接图失败 {concat_path}: {e}")
        if concat_img is None:
            concat_img = concat_images(path_a, path_b)

        if concat_img is not None:
            marked = draw_mark(concat_img, human_label, model_is_consistent)
            try:
                marked.save(out_img_path, quality=85)
            except Exception as e:
                log(f"[Warn] 写入拼接图失败 {out_img_path}: {e}")

    log(f"[Summary] 一致: {agree_true} 张，不一致: {agree_false} 张，拒绝采样: {reject_cnt} 张")


if __name__ == "__main__":
    main()

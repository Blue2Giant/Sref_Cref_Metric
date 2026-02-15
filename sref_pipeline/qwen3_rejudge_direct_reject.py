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
from urllib.parse import urlparse, parse_qs, unquote

API_KEY = "EMPTY"
MODEL = "Qwen3VL30BA3B-Image-Edit"
BASE_URL = "http://stepcast-router.shai-core:9200/v1"
TIMEOUT = 180

RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85

# “单次判别有效”的置信度阈值：conf > 该值 才算有效
JUDGE_CONF_THRESHOLD = 0.5

# 多次判别参数：判别 5 次，至少 4 次有效才接受
JUDGE_TIMES = 5
JUDGE_MIN_VALID = 4

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
    "请仅从“画风 / 视觉风格”的角度比较图片 A 和图片 B，忽略画面中的人物身份、动作含义和故事语义，只关注视觉表现形式。\n\n"
    "你需要从以下维度综合判断两张图片的画风是否一致：\n"
    "1. 纹理与材质：画面的质感（如粗糙、细腻、油画感、水彩感、数码绘图感、胶片颗粒感等）。\n"
    "2. 色彩运用：整体色调（冷暖、饱和度、对比度）、特定的配色方案或色彩倾向。\n"
    "3. 笔触与线条：线条的粗细、锐利度、笔触的可见性、描边风格（如无描边、粗黑边等）。\n"
    "4. 光影处理：光照来源、阴影强度、立体感渲染方式（如二次元平涂、厚涂、真实光影等）。\n"
    "5. 几何构造与形态：主体的造型特征（如写实、夸张、Q版/Chibi、极简、抽象等）。如果是 Q 版，请关注头身比、面部特征等是否一致。\n"
    "6. 构图与视角：画面的布局方式、透视感（如正视、俯视、仰视、鱼眼等）。\n\n"
    "在做出判断时，请特别关注几何构造与形态、光影处理以及笔触与线条是否一致；如果在大部分关键维度上高度一致，则认为画风一致，输出 1；只要有明显差异，则认为画风不一致，输出 0。\n"
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
    if x > 60:
        return math.exp(60)
    if x < -60:
        return math.exp(-60)
    return math.exp(x)


def _softmax2(logp0: float, logp1: float) -> Tuple[float, float]:
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

    if need_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(top_logprobs)
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
    mapping: Dict[str, float] = {}

    if not resp_json:
        return mapping
    choices = resp_json.get("choices", [])
    if not choices:
        return mapping
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    logprobs = choice0.get("logprobs", None)

    if logprobs is None:
        msg = choice0.get("message", {}) if isinstance(choice0.get("message"), dict) else {}
        logprobs = msg.get("logprobs", None)

    if not isinstance(logprobs, dict):
        return mapping

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
                for tok, lp in top.items():
                    if isinstance(tok, str) and isinstance(lp, (int, float)):
                        mapping[tok] = float(lp)
                if mapping:
                    return mapping

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
            for item in first:
                if isinstance(item, dict):
                    tok = item.get("token")
                    lp = item.get("logprob")
                    if isinstance(tok, str) and isinstance(lp, (int, float)):
                        mapping[tok] = float(lp)
            if mapping:
                return mapping

    return mapping


def _extract_01_logprobs(resp_json: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
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
    单次判别：返回 (label_or_None, reason, confidence_or_None)
    注意：这里不再因为置信度低而 reject；是否“有效”交给多次判别逻辑决定。
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
        return None, "API 无响应/请求失败", None

    choices = resp_json.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return None, "返回结构异常(无 choices)", None

    text = strip_code_fences(_extract_text_from_choice(choices[0])).strip()
    pred_char = None
    for ch in text:
        if not ch.isspace():
            pred_char = ch
            break

    if pred_char not in ("0", "1"):
        return None, f"输出不是 0/1 (got={text!r})", None

    pred_is_consistent = (pred_char == "1")

    logp0, logp1, mapping_raw = _extract_01_logprobs(resp_json)
    if logp0 is None or logp1 is None:
        return None, f"无法提取 0/1 top_logprobs (keys={list(mapping_raw.keys())[:8]})", None

    p0, p1 = _softmax2(logp0, logp1)
    conf = p1 if pred_is_consistent else p0

    reason = f"pred={pred_char}, conf={conf:.3f} (p0={p0:.3f}, p1={p1:.3f})"
    return pred_is_consistent, reason, conf


def multi_judge_images_01(
    path_a: str,
    path_b: str,
    times: int = 5,
    min_valid: int = 4,
    valid_conf_threshold: float = 0.5,
) -> Tuple[Optional[bool], str, Optional[float], Dict[str, Any]]:
    """
    多次判别：
      - 判别 times 次
      - 每次 conf > valid_conf_threshold 才算“有效”
      - 有效次数 >= min_valid 才接受，否则整体 reject
      - 接受后对“有效判别”多数投票得到最终 label
    """
    trials: List[Dict[str, Any]] = []
    for t in range(times):
        pred, reason, conf = direct_judge_images_01(path_a, path_b)
        is_valid = isinstance(pred, bool) and isinstance(conf, (int, float)) and (conf > valid_conf_threshold)
        trials.append(
            {
                "trial": t,
                "pred": pred,
                "conf": conf,
                "valid": is_valid,
                "reason": reason,
            }
        )

    valid_trials = [x for x in trials if x.get("valid") is True]
    n_valid = len(valid_trials)

    detail: Dict[str, Any] = {
        "times": times,
        "min_valid": min_valid,
        "valid_conf_threshold": valid_conf_threshold,
        "valid_trials": n_valid,
        "trials": trials,
    }

    if n_valid < min_valid:
        reason = f"reject: valid_trials={n_valid}/{times} (<{min_valid}), thr>{valid_conf_threshold}"
        return None, reason, None, detail

    n1 = sum(1 for x in valid_trials if x["pred"] is True)
    n0 = n_valid - n1

    if n1 > n0:
        final = True
    elif n0 > n1:
        final = False
    else:
        # 平票：用置信度总和打破
        sum1 = sum(float(x["conf"]) for x in valid_trials if x["pred"] is True and x["conf"] is not None)
        sum0 = sum(float(x["conf"]) for x in valid_trials if x["pred"] is False and x["conf"] is not None)
        final = (sum1 >= sum0)

    chosen_confs = [float(x["conf"]) for x in valid_trials if x["pred"] == final and x["conf"] is not None]
    overall_conf = (sum(chosen_confs) / len(chosen_confs)) if chosen_confs else None

    reason = (
        f"accept: valid_trials={n_valid}/{times} (thr>{valid_conf_threshold}), "
        f"vote:1={n1},0={n0}, final={'1' if final else '0'}, "
        f"mean_conf={overall_conf:.3f}" if overall_conf is not None else
        f"accept: valid_trials={n_valid}/{times} (thr>{valid_conf_threshold}), "
        f"vote:1={n1},0={n0}, final={'1' if final else '0'}"
    )

    detail["vote_1"] = n1
    detail["vote_0"] = n0
    detail["final"] = final
    detail["overall_conf"] = overall_conf

    return final, reason, overall_conf, detail


def main():
    global MODEL, BASE_URL, JUDGE_CONF_THRESHOLD, JUDGE_TIMES, JUDGE_MIN_VALID

    parser = argparse.ArgumentParser(description="用 0/1 单 token 判别风格一致性，并用 0/1 token logprob 计算置信度（多次投票）")
    parser.add_argument("--image_txt", required=True, help="包含图片完整路径的txt文件")
    parser.add_argument("--json_dir", required=True, help="原始结果JSON所在目录")
    parser.add_argument("--output_dir", required=True, help="新的结果输出目录")
    parser.add_argument("--model", type=str, default=MODEL, help="Qwen 模型名称")
    parser.add_argument("--base_url", type=str, default=BASE_URL, help="Qwen 接口 base url")

    # 这里的 conf_threshold 现在表示：单次判别有效的阈值（默认 0.5）
    parser.add_argument("--conf_threshold", type=float, default=JUDGE_CONF_THRESHOLD, help="单次有效置信度阈值(conf>该值才算有效)")
    parser.add_argument("--judge_times", type=int, default=JUDGE_TIMES, help="同一对图判别次数（默认5）")
    parser.add_argument("--min_valid", type=int, default=JUDGE_MIN_VALID, help="至少多少次有效才接受（默认4）")

    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在的输出文件")
    args = parser.parse_args()

    MODEL = args.model
    BASE_URL = args.base_url
    JUDGE_CONF_THRESHOLD = float(args.conf_threshold)
    JUDGE_TIMES = int(args.judge_times)
    JUDGE_MIN_VALID = int(args.min_valid)

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

        model_is_consistent, model_reason, model_conf, judge_detail = multi_judge_images_01(
            path_a,
            path_b,
            times=JUDGE_TIMES,
            min_valid=JUDGE_MIN_VALID,
            valid_conf_threshold=JUDGE_CONF_THRESHOLD,
        )

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
            new_data["is_consistent_new"] = "reject"

        new_data["reason_new"] = model_reason
        new_data["confidence"] = model_conf
        new_data["agree"] = agree

        # 记录 5 次判别的细节，方便你回溯
        new_data["judge_detail"] = judge_detail

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

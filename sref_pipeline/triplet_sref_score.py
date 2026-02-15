#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import base64
import argparse
import math
from io import BytesIO
from typing import Dict, Optional, List, Tuple, Any

import requests
from PIL import Image

from megfile.smart import (
    smart_listdir,
    smart_exists,
    smart_makedirs,
    smart_open as mopen,
)
"""
python judge_triplets_style.py \
  --root s3://lanjinghong-data/loras_triplets/flux_0111_triplets_all \
  --num_samples 200 \
  --seed 42 \
  --out_json s3://lanjinghong-data/loras_triplets/flux_0111_triplets_all/style_judge_200.json
"""

# ================== API 配置 ==================
API_KEY = "EMPTY"
MODEL = "Qwen3VL30BA3B-Image-Edit"
BASE_URL = "http://stepcast-router.shai-core:9200/v1"
TIMEOUT = 180

# ================== 图片处理 ==================
RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85
Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

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
    "1. 纹理与材质\n"
    "2. 色彩运用\n"
    "3. 笔触与线条\n"
    "4. 光影处理\n"
    "5. 几何构造与形态\n"
    "6. 构图与视角\n\n"
    "如果在大部分关键维度上高度一致，则认为画风一致，输出 1；只要有明显差异，则认为画风不一致，输出 0。\n"
    "最终只输出一个字符：0 或 1。"
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def norm_dir(p: str) -> str:
    return p if p.endswith("/") else (p + "/")


def join_path(base: str, name: str) -> str:
    return base + name if base.endswith("/") else base + "/" + name


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        if path.startswith("s3://"):
            with mopen(path, "rb") as f:
                return f.read()
        else:
            with open(path, "rb") as f:
                return f.read()
    except Exception as e:
        log(f"[Warn] 读取失败 {path}: {e}")
        return None


def _load_image(path: str) -> Optional[Image.Image]:
    b = _read_bytes(path)
    if b is None:
        return None
    try:
        img = Image.open(BytesIO(b))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        log(f"[Warn] 解码图片失败 {path}: {e}")
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
    单次判别：返回 (pred_or_None, reason, conf_or_None)
    pred=True 表示输出1（风格一致）
    conf 是对当前输出类别的置信度（基于 0/1 logprob softmax）
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


def judge_with_reject_sampling(
    fusion_path: str,
    style_path: str,
    conf_thr: float,
    target_valid: int,
    min_true: int,
    max_calls: int,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """
    拒绝采样版本：
    - 单次 conf > conf_thr 才计入 valid
    - 收集到 target_valid 条 valid（或达到 max_calls 次）停止
    - 单个 style 判定 True：valid 中至少 min_true 次满足 (pred==True 且 conf>conf_thr)
    - 若 valid 不足 -> 返回 None（reject）
    """
    trials: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []

    calls = 0
    while calls < max_calls and len(valid) < target_valid:
        calls += 1
        pred, reason, conf = direct_judge_images_01(fusion_path, style_path)

        is_valid = isinstance(pred, bool) and isinstance(conf, (int, float)) and (conf > conf_thr)
        rec = {
            "call": calls,
            "pred": pred,
            "conf": conf,
            "valid": is_valid,
            "reason": reason,
        }
        trials.append(rec)
        if is_valid:
            valid.append(rec)

    detail = {
        "fusion": fusion_path,
        "style": style_path,
        "conf_thr": conf_thr,
        "target_valid": target_valid,
        "min_true": min_true,
        "max_calls": max_calls,
        "calls": calls,
        "valid_n": len(valid),
        "trials": trials,
    }

    if len(valid) < target_valid:
        detail["status"] = "reject"
        return None, detail

    true_good = sum(1 for x in valid if x["pred"] is True and x["conf"] is not None and x["conf"] > conf_thr)
    passed = (true_good >= min_true)
    detail["status"] = "ok"
    detail["true_good"] = true_good
    detail["passed"] = passed
    return passed, detail


def smart_write_json(path: str, obj: Any):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    if path.startswith("s3://"):
        with mopen(path, "wb") as f:
            f.write(data)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def list_style_dirs(root: str) -> Tuple[str, List[str]]:
    """
    返回：
      fusion_dir (style_and_content/)
      style_dirs  (style_1/, style_2/, ... 按编号排序)
    """
    root = norm_dir(root)
    fusion_dir = join_path(root, "style_and_content/")
    if not smart_exists(fusion_dir):
        raise RuntimeError(f"缺少目录: {fusion_dir}")

    entries = smart_listdir(root)
    style_dirs = []
    for e in entries:
        if not e.endswith("/"):
            continue
        if re.fullmatch(r"style_\d+/", e):
            style_dirs.append(e)

    def key_fn(x: str) -> int:
        m = re.search(r"style_(\d+)/", x)
        return int(m.group(1)) if m else 10**9

    style_dirs.sort(key=key_fn)
    return fusion_dir, [join_path(root, d) for d in style_dirs]


def list_candidate_names(fusion_dir: str) -> List[str]:
    names = []
    for e in smart_listdir(fusion_dir):
        if e.endswith("/"):
            continue
        if is_image_name(e):
            names.append(e)
    names.sort()
    return names


def main():
    global MODEL, BASE_URL

    ap = argparse.ArgumentParser("从 triplets 输出目录随机抽样，按拒绝采样策略判定画风相似度")
    ap.add_argument("--root", required=True, help="输出目录根：包含 style_and_content/ style_1/ style_2/... content_*/")
    ap.add_argument("--num_samples", type=int, default=100, help="随机抽样数量（<=0 表示全量）")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--conf_thr", type=float, default=0.5, help="单次有效阈值（conf>thr 才算 valid）")

    ap.add_argument("--target_valid", type=int, default=3, help="每个 style_i 需要收集的 valid 次数")
    ap.add_argument("--min_true", type=int, default=2, help="valid 中至少多少次 (pred=1 且 conf>thr) 才算单个 style_i 通过")
    ap.add_argument("--max_calls", type=int, default=10, help="为凑够 target_valid 最多调用次数（拒绝采样上限）")

    ap.add_argument("--style_ratio", type=float, default=0.6, help=">= 该比例的 style_i 通过，整体判定相似")
    ap.add_argument("--repeat_only_style1", action="store_true",
                    help="若开启：只有 style_1 用 target_valid/min_true，其它 style_i 只判 1 次（无拒绝采样）")

    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--base_url", type=str, default=BASE_URL)

    ap.add_argument("--out_json", required=True, help="结果输出 JSON（可写 s3:// 或本地路径）")
    args = ap.parse_args()

    MODEL = args.model
    BASE_URL = args.base_url

    fusion_dir, style_dirs = list_style_dirs(args.root)
    if not style_dirs:
        raise RuntimeError(f"在 {args.root} 下没找到 style_*/ 目录")

    cand_names = list_candidate_names(fusion_dir)
    if not cand_names:
        raise RuntimeError(f"{fusion_dir} 下没找到图片")

    import random
    rng = random.Random(args.seed)
    if args.num_samples <= 0 or args.num_samples >= len(cand_names):
        picked = cand_names
    else:
        picked = rng.sample(cand_names, args.num_samples)

    log(f"[Info] root={args.root}")
    log(f"[Info] fusion_dir={fusion_dir}")
    log(f"[Info] style_dirs={len(style_dirs)} -> {', '.join([os.path.basename(d.rstrip('/')) for d in style_dirs])}")
    log(f"[Info] candidates={len(cand_names)}, picked={len(picked)}")

    results: List[Dict[str, Any]] = []
    ok_cnt = 0

    for i, name in enumerate(picked, 1):
        fusion_path = join_path(fusion_dir, name)

        per_style: List[Dict[str, Any]] = []
        passed_style = 0
        total_style = len(style_dirs)

        for sdir in style_dirs:
            style_path = join_path(sdir, name)
            style_tag = os.path.basename(sdir.rstrip("/"))  # style_1

            if not smart_exists(style_path):
                per_style.append({
                    "style_dir": style_tag,
                    "exists": False,
                    "decision": False,
                    "detail": {"status": "missing_file", "style": style_path}
                })
                continue

            # 如果只让 style_1 重复，其它只判一次
            if args.repeat_only_style1 and style_tag != "style_1":
                pred, reason, conf = direct_judge_images_01(fusion_path, style_path)
                decision = bool(pred is True and conf is not None and conf > args.conf_thr)
                detail = {
                    "status": "single",
                    "pred": pred,
                    "conf": conf,
                    "reason": reason,
                    "conf_thr": args.conf_thr,
                }
            else:
                decision, detail = judge_with_reject_sampling(
                    fusion_path=fusion_path,
                    style_path=style_path,
                    conf_thr=args.conf_thr,
                    target_valid=args.target_valid,
                    min_true=args.min_true,
                    max_calls=args.max_calls,
                )
                # decision=None 表示 reject，按 fail 计
                decision = bool(decision is True)

            if decision:
                passed_style += 1

            per_style.append({
                "style_dir": style_tag,
                "exists": True,
                "decision": decision,
                "detail": detail,
            })

        ratio = passed_style / float(total_style) if total_style > 0 else 0.0
        overall = (ratio >= args.style_ratio)
        if overall:
            ok_cnt += 1

        rec = {
            "name": name,
            "fusion": fusion_path,
            "passed_style": passed_style,
            "total_style": total_style,
            "ratio": ratio,
            "style_ratio_thr": args.style_ratio,
            "overall_similar": overall,
            "per_style": per_style,
        }
        results.append(rec)

        log(f"[{i}/{len(picked)}] {name} -> passed {passed_style}/{total_style} ({ratio:.3f}) overall={overall}")

    summary = {
        "root": args.root,
        "picked": len(picked),
        "overall_true": ok_cnt,
        "overall_ratio": ok_cnt / float(len(picked)) if picked else 0.0,
        "conf_thr": args.conf_thr,
        "target_valid": args.target_valid,
        "min_true": args.min_true,
        "max_calls": args.max_calls,
        "style_ratio_thr": args.style_ratio,
        "repeat_only_style1": bool(args.repeat_only_style1),
        "model": MODEL,
        "base_url": BASE_URL,
    }

    out_obj = {"summary": summary, "results": results}

    # 输出目录（本地时才需要创建）
    if not args.out_json.startswith("s3://"):
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    smart_write_json(args.out_json, out_obj)
    log(f"[DONE] overall_true={ok_cnt}/{len(picked)} ({summary['overall_ratio']:.3f}) -> {args.out_json}")


if __name__ == "__main__":
    main()

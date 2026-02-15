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
import multiprocessing as mp
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

API_KEY = "EMPTY"
MODEL = "Qwen3VL30BA3B-Image-Edit"
MODEL = "v1p3"
BASE_URL = "http://stepcast-router.shai-core:9200/v1"
TIMEOUT = 180

RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85
Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

SYSTEM_PROMPT_01 = (
    "你是一个只关注“主体内容和主题”的严格评审。\n"
    "你必须完全忽略画风、渲染风格、分辨率、滤镜等视觉风格差异，\n"
    "只关心画面中出现的具体人物/物体/场景是否相同或高度一致。\n"
    "你的任务：判断两张图片在“主体内容/主题”上是否高度一致。\n"
    "你必须只输出一个字符：0 或 1。\n"
    "1 表示主体内容高度一致；0 表示主体内容不一致。\n"
    "不要输出任何多余文字、空格、换行、JSON。"
)

USER_INSTRUCTION_01 = (
    "请只从“主体内容和主题”的角度比较图片 A 和图片 B，\n"
    "严格忽略画风、线条风格、色彩风格、渲染方式等一切视觉风格差异。\n\n"
    "你需要重点考虑：画面里“是什么”和“在做什么”，而不是“怎么画出来的”。\n\n"
    "具体判定规则如下：\n"
    "1. 如果是【人物】为主体：\n"
    "   - 关注人物是否为“同一个角色”或“极为相似的角色”，\n"
    "   - 包括性别、年龄段、身材、发型、头发颜色、肤色、服装类型、服装主色调、主要配饰等是否相近，\n"
    "   - 姿势、朝向、镜头视角可以有一定变化，但如果感觉明显是不同的人物或完全不同造型，则视为不一致。\n"
    "2. 如果是【单一物体】为主体（例如一辆车、一栋房子、一把椅子等）：\n"
    "   - 重点看物体类别和形状结构是否一致（例如都是跑车、都是 SUV、都是圆桌等），\n"
    "   - 允许颜色不同，例如黄色的车和红色的车，只要车型和外形高度相似，就可以视为一致，\n"
    "   - 如果只是都包含“车/房子/杯子”但明显是不同种类或完全不同结构，则视为不一致。\n"
    "3. 如果是【复杂场景】（例如街景、室内布景、多人物场景等）：\n"
    "   - 关注场景的类型、主要元素组合和布局是否相似，\n"
    "   - 例如：都为“一个人物站在城市夜景街道中央，背景有高楼和霓虹招牌”，可以视为一致，\n"
    "   - 如果只是都在室外/室内，但核心构图和主体物体完全不同，则视为不一致。\n"
    "4. 画风完全无关：\n"
    "   - 即使一张是写实照片，另一张是二次元插画或卡通，只要主体内容和主题高度一致，也要判为 1，\n"
    "   - 禁止因为画风差异而判为 0。\n\n"
    "综合以上规则：\n"
    "当你认为两张图展示的是“同一个人物/同一类具体物体/同一种具体场景和主题”，则输出 1；\n"
    "如果只是大概类别相似（例如都有人物/都有车），但主体明显不是同一个，就输出 0。\n\n"
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
                {"type": "text", "text": "下面是图片 A（triplet 图）："},
                {"type": "image_url", "image_url": {"url": data_a}},
                {"type": "text", "text": "下面是图片 B（content_* 中的参考图）："},
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

    pred_is_consistent = pred_char == "1"

    logp0, logp1, mapping_raw = _extract_01_logprobs(resp_json)
    if logp0 is None or logp1 is None:
        return None, f"无法提取 0/1 top_logprobs (keys={list(mapping_raw.keys())[:8]})", None

    p0, p1 = _softmax2(logp0, logp1)
    conf = p1 if pred_is_consistent else p0
    reason = f"pred={pred_char}, conf={conf:.3f} (p0={p0:.3f}, p1={p1:.3f})"
    return pred_is_consistent, reason, conf


def judge_subject_with_votes(
    triplet_path: str,
    content_path: str,
    conf_thr: float,
    judge_times: int,
    min_true: int,
) -> Tuple[bool, Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    good_true = 0

    for i in range(1, judge_times + 1):
        pred, reason, conf = direct_judge_images_01(triplet_path, content_path)
        is_valid = isinstance(pred, bool) and isinstance(conf, (int, float)) and (conf > conf_thr)
        if is_valid and pred is True:
            good_true += 1
        trials.append(
            {
                "call": i,
                "pred": pred,
                "conf": conf,
                "valid": is_valid,
                "reason": reason,
            }
        )

    passed = good_true >= min_true
    detail = {
        "triplet": triplet_path,
        "content": content_path,
        "conf_thr": conf_thr,
        "judge_times": judge_times,
        "min_true": min_true,
        "good_true": good_true,
        "trials": trials,
        "status": "ok",
        "passed": passed,
    }
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


def list_content_dirs(root: str) -> Tuple[str, List[str]]:
    root = norm_dir(root)
    triplet_dir = join_path(root, "style_and_content/")
    if not smart_exists(triplet_dir):
        raise RuntimeError(f"缺少目录: {triplet_dir}")

    entries = smart_listdir(root)
    content_dirs = []
    for e in entries:
        name = str(e).rstrip("/")
        if re.fullmatch(r"content_\d+", name):
            content_dirs.append(name + "/")

    def key_fn(x: str) -> int:
        m = re.search(r"content_(\d+)/", x)
        return int(m.group(1)) if m else 10**9

    content_dirs.sort(key=key_fn)
    return triplet_dir, [join_path(root, d) for d in content_dirs]


def list_candidate_names(triplet_dir: str) -> List[str]:
    names = []
    for e in smart_listdir(triplet_dir):
        if e.endswith("/"):
            continue
        if is_image_name(e):
            names.append(e)
    names.sort()
    return names


def main():
    global MODEL, BASE_URL

    ap = argparse.ArgumentParser("从 triplets 输出目录随机抽样，按多次 0/1 判别主体内容一致性")
    ap.add_argument("--root", required=True, help="输出目录根：包含 style_and_content/ content_1/ content_2/...")
    ap.add_argument("--num_samples", type=int, default=100, help="随机抽样数量（<=0 表示全量）")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--conf_thr", type=float, default=0.5, help="单次有效阈值（conf>thr 才算有效投票）")
    ap.add_argument("--judge_times", type=int, default=3, help="同一对 (triplet, content) 判别次数")
    ap.add_argument("--min_true", type=int, default=2, help="至少多少次 (pred=1 且 conf>thr) 才判该 content_i 为通过")

    ap.add_argument("--content_ratio", type=float, default=0.6, help=">= 该比例的 content_i 通过，整体判定主体内容一致")

    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--base_url", type=str, default=BASE_URL)

    ap.add_argument("--out_all", required=True, help="全量 map json：basename_no_ext -> 1/0（可写 s3:// 或本地）")
    ap.add_argument("--out_pos", required=True, help="正样本 map json：只含 value=1 的键")
    ap.add_argument("--out_neg", required=True, help="负样本 map json：只含 value=0 的键")

    ap.add_argument("--out_detail", default="", help="（可选）写详细结果 JSON")
    ap.add_argument("--num_procs", type=int, default=0, help="多进程 worker 数；0=自动(按CPU核数)")

    args = ap.parse_args()

    MODEL = args.model
    BASE_URL = args.base_url

    triplet_dir, content_dirs = list_content_dirs(args.root)
    if not content_dirs:
        raise RuntimeError(f"在 {args.root} 下没找到 content_*/ 目录")

    cand_names = list_candidate_names(triplet_dir)
    if not cand_names:
        raise RuntimeError(f"{triplet_dir} 下没找到图片")

    import random

    rng = random.Random(args.seed)
    if args.num_samples <= 0 or args.num_samples >= len(cand_names):
        picked = cand_names
    else:
        picked = rng.sample(cand_names, args.num_samples)

    log(f"[Info] picked={len(picked)} from candidates={len(cand_names)}")

    def _process_one_name(name: str) -> Tuple[Dict[str, Any], int]:
        triplet_path = join_path(triplet_dir, name)

        per_content: List[Dict[str, Any]] = []
        passed_content = 0
        total_content = len(content_dirs)

        for cdir in content_dirs:
            content_path = join_path(cdir, name)
            content_tag = os.path.basename(cdir.rstrip("/"))

            if not smart_exists(content_path):
                per_content.append(
                    {
                        "content_dir": content_tag,
                        "exists": False,
                        "decision": False,
                        "detail": {"status": "missing_file", "content": content_path},
                    }
                )
                continue

            decision, detail = judge_subject_with_votes(
                triplet_path=triplet_path,
                content_path=content_path,
                conf_thr=args.conf_thr,
                judge_times=args.judge_times,
                min_true=args.min_true,
            )
            decision = bool(decision is True)

            if decision:
                passed_content += 1

            per_content.append(
                {
                    "content_dir": content_tag,
                    "exists": True,
                    "decision": decision,
                    "detail": detail,
                }
            )

        ratio = passed_content / float(total_content) if total_content > 0 else 0.0
        overall = ratio >= args.content_ratio
        ok_inc = 1 if overall else 0

        rec = {
            "name": name,
            "triplet": triplet_path,
            "passed_content": passed_content,
            "total_content": total_content,
            "ratio": ratio,
            "content_ratio_thr": args.content_ratio,
            "overall_consistent": overall,
            "per_content": per_content,
        }
        return rec, ok_inc

    results: List[Dict[str, Any]] = []
    ok_cnt = 0

    if args.num_procs and args.num_procs > 1:
        procs = args.num_procs
        if procs <= 0:
            procs = mp.cpu_count() or 1
        procs = max(1, procs)
        log(f"[Info] 使用多进程 num_procs={procs}")
        with mp.Pool(processes=procs) as pool:
            for rec, ok_inc in pool.map(_process_one_name, picked):
                results.append(rec)
                ok_cnt += ok_inc
    else:
        for i, name in enumerate(picked, 1):
            rec, ok_inc = _process_one_name(name)
            results.append(rec)
            ok_cnt += ok_inc
            log(
                f"[{i}/{len(picked)}] {name} -> {rec['passed_content']}/{rec['total_content']} "
                f"ratio={rec['ratio']:.3f} overall={rec['overall_consistent']}"
            )

    all_map: Dict[str, int] = {}
    pos_map: Dict[str, int] = {}
    neg_map: Dict[str, int] = {}

    for rec in results:
        fname = rec["name"]
        base = os.path.splitext(fname)[0]
        v = 1 if rec.get("overall_consistent") else 0
        all_map[base] = v
        if v == 1:
            pos_map[base] = 1
        else:
            neg_map[base] = 0

    smart_write_json(args.out_all, all_map)
    smart_write_json(args.out_pos, pos_map)
    smart_write_json(args.out_neg, neg_map)

    log(f"[DONE] all={len(all_map)} pos={len(pos_map)} neg={len(neg_map)}")
    log(f"  -> {args.out_all}")
    log(f"  -> {args.out_pos}")
    log(f"  -> {args.out_neg}")

    if args.out_detail:
        summary = {
            "root": args.root,
            "picked": len(picked),
            "overall_true": ok_cnt,
            "overall_ratio": ok_cnt / float(len(picked)) if picked else 0.0,
            "conf_thr": args.conf_thr,
            "judge_times": args.judge_times,
            "min_true": args.min_true,
            "content_ratio_thr": args.content_ratio,
            "model": MODEL,
            "base_url": BASE_URL,
        }
        smart_write_json(args.out_detail, {"summary": summary, "results": results})
        log(f"[Detail] -> {args.out_detail}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import base64
import json
import math
import multiprocessing as mp
import os
import queue
import random
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm
from megfile.smart import smart_exists


G_TASKS: List[Dict[str, Any]] = []
G_ARGS = None
API_KEY = "EMPTY"
MODEL = "Qwen3-VL-30B-A3B-Instruct"
BASE_URL = "http://10.201.19.61:22002/v1"
TIMEOUT = 720
RETRY_EXHAUSTED_REASON = "API 重试耗尽"
RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85
Image.MAX_IMAGE_PIXELS = None
STYLE_SYSTEM_PROMPT = (
    "你是一个只关注“画风/视觉风格”的资深评审。\n"
    "你只评估视觉表现形式（媒介感、材质感、线条/笔触、色彩与调色、光影与对比、渲染/后期、画面噪声与颗粒、细节表达方式）。\n"
    "你必须忽略：人物/物体身份、动作含义、故事语义、场景类别、构图内容是否相似。\n"
    "\n"
    "判定目标：两张图是否属于同一种稳定画风/同一风格族。\n"
    "允许以下差异仍判为风格一致：\n"
    "- 内容/主体/场景不同\n"
    "- 构图与视角不同\n"
    "- 色相轻微变化、亮度对比变化、局部调色差异\n"
    "- 细节密度不同、裁剪/分辨率不同、轻微压缩/噪声\n"
    "\n"
    "只有当出现“风格机制”层面的明显变化才判不一致，例如：\n"
    "- 真实摄影 vs 插画/渲染\n"
    "- 线稿/勾线体系变化（有线稿→无，粗线→细线，漫画勾线→水彩边缘）\n"
    "- 材质与纹理生成方式变化（油画厚涂→平涂赛璐璐→3D塑料感→像素/点描等）\n"
    "- 光影模型变化（硬边影视布光→柔和漫反射插画光→霓虹强对比等）\n"
    "- 调色与色彩策略变化（低饱和复古→高饱和糖果色→黑白素描等）\n"
    "\n"
    "输出规则：你只能输出一个字符：0 或 1。\n"
    "1 = 画风高度一致（同一风格族，核心机制一致）；0 = 画风不一致。\n"
    "不要输出任何多余文字、空格、换行或标点。"
)
STYLE_USER_INSTRUCTION = (
    "请仅从“画风 / 视觉风格”角度比较图片A与图片B，忽略人物/物体身份、动作含义、故事语义与场景类别。\n"
    "\n"
    "请综合以下维度做判断，并采用“宽松一致性”标准：只要核心风格机制一致，即使主体、构图、视角、细节密度不同，也可以判为一致。\n"
    "重点维度（更高权重）：\n"
    "1) 媒介与渲染方式：摄影/3D/插画/水彩/油画/厚涂/赛璐璐/像素/素描 等\n"
    "2) 笔触与线条体系：是否有线稿、线条粗细/抖动、边缘处理、笔触颗粒\n"
    "3) 材质与纹理生成方式：表面质感、噪声/颗粒、细节组织方式\n"
    "4) 光影模型与对比：硬/软阴影、体积光、漫反射/镜面、高反差与否\n"
    "5) 色彩策略：饱和度、色相偏好、综合色调、调色风格（复古/冷暖/霓虹等）\n"
    "次要维度（允许变化）：\n"
    "6) 构图与视角：机位、镜头感、取景范围不同不应直接判为不一致\n"
    "\n"
    "判定：\n"
    "- 若多数“重点维度”一致，输出 1。\n"
    "- 只要出现明显的风格机制改变（如摄影↔插画、线稿体系突变、材质/渲染范式突变、整体调色策略完全不同），输出 0。\n"
    "\n"
    "最终只输出一个字符：0 或 1。"
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
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
    except Exception:
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


def call_qwen_chat_with_retry(
    messages: list,
    temperature: float,
    max_tokens: int,
    need_logprobs: bool,
    top_logprobs: int,
    retry_times: int,
    retry_delay: float,
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
    total_attempts = max(0, int(retry_times)) + 1
    for attempt in range(1, total_attempts + 1):
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
            log(f"[Err] API 请求出错(第 {attempt}/{total_attempts} 次): {e}")
            if attempt < total_attempts:
                time.sleep(retry_delay)
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
    choices = resp_json.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return mapping
    choice0 = choices[0]
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


def direct_judge_images_generic(path_a: str, path_b: str, system_prompt: str, user_instruction: str) -> Tuple[Optional[bool], str, Optional[float]]:
    data_a = get_image_data_uri(path_a)
    data_b = get_image_data_uri(path_b)
    if not data_a or not data_b:
        return None, "图片编码失败", None
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instruction},
                {"type": "text", "text": "下面是图片 A："},
                {"type": "image_url", "image_url": {"url": data_a}},
                {"type": "text", "text": "下面是图片 B："},
                {"type": "image_url", "image_url": {"url": data_b}},
                {"type": "text", "text": "只输出一个字符：0 或 1。"},
            ],
        },
    ]
    args = G_ARGS
    retry_times = int(args.conn_retry_times) if args is not None else 0
    retry_delay = float(args.conn_retry_delay) if args is not None else 2.0
    resp_json = call_qwen_chat_with_retry(
        messages,
        temperature=0.0,
        max_tokens=1,
        need_logprobs=True,
        top_logprobs=8,
        retry_times=retry_times,
        retry_delay=retry_delay,
    )
    if not resp_json:
        return None, RETRY_EXHAUSTED_REASON, None
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


def judge_pair_voting(
    path_a: str,
    path_b: str,
    system_prompt: str,
    user_instruction: str,
    conf_thr: float,
    judge_times: int,
    min_true: int,
) -> Tuple[Optional[bool], Dict[str, Any], bool]:
    trials: List[Dict[str, Any]] = []
    good_true = 0
    retry_exhausted = False
    for i in range(1, int(judge_times) + 1):
        pred, reason, conf = direct_judge_images_generic(path_a, path_b, system_prompt, user_instruction)
        if pred is None and reason == RETRY_EXHAUSTED_REASON:
            retry_exhausted = True
            break
        is_valid = isinstance(pred, bool) and isinstance(conf, (int, float)) and (float(conf) > float(conf_thr))
        if is_valid and pred is True:
            good_true += 1
        trials.append({"call": i, "pred": pred, "conf": conf, "valid": is_valid, "reason": reason})
    if retry_exhausted:
        return None, {"status": "retry_exhausted", "trials": trials, "good_true": good_true}, True
    passed = good_true >= int(min_true)
    return passed, {"status": "ok", "trials": trials, "good_true": good_true, "passed": passed}, False


def read_style_index(path: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                if not isinstance(k, str):
                    continue
                if not isinstance(v, list):
                    continue
                vals = [str(x).strip() for x in v if isinstance(x, str) and str(x).strip()]
                if vals:
                    index[k] = vals
    return index


def parse_triplet_jsonl(path: str, style_index: Dict[str, List[str]]) -> Tuple[List[Dict[str, Any]], int]:
    tasks: List[Dict[str, Any]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                skipped += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            for pair_key, arr in obj.items():
                if not isinstance(pair_key, str):
                    skipped += 1
                    continue
                if not isinstance(arr, list) or not arr:
                    skipped += 1
                    continue
                main_img = str(arr[0]).strip()
                if not main_img:
                    skipped += 1
                    continue
                if "__" not in pair_key:
                    skipped += 1
                    continue
                content_id, style_id = pair_key.split("__", 1)
                style_id = style_id.strip()
                style_imgs = style_index.get(style_id, [])
                tasks.append(
                    {
                        "pair_key": pair_key,
                        "content_id": content_id.strip(),
                        "style_id": style_id,
                        "main_img": main_img,
                        "style_imgs": style_imgs,
                    }
                )
    return tasks, skipped


def load_existing_done_keys(out_jsonl: str) -> Dict[str, int]:
    done: Dict[str, int] = {}
    if not out_jsonl or not os.path.isfile(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                if isinstance(k, str) and v in (0, 1):
                    done[k] = int(v)
    return done


def _judge_one(task: Dict[str, Any]) -> Dict[str, Any]:
    args = G_ARGS
    pair_key = task["pair_key"]
    main_img = task["main_img"]
    style_imgs = task["style_imgs"]

    if not smart_exists(main_img):
        return {"pair_key": pair_key, "result": None, "error": f"main_not_found: {main_img}", "detail": {}}
    if not style_imgs:
        return {"pair_key": pair_key, "result": None, "error": f"style_id_not_found: {task['style_id']}", "detail": {}}
    passed_style = 0
    total_style = 0
    per_style_details = []
    for i, sp in enumerate(style_imgs):
        tag = f"style_{i+1}"
        if not smart_exists(sp):
            per_style_details.append({"dir": tag, "exists": False})
            continue
        total_style += 1
        if bool(args.style_repeat_only_style1) and tag != "style_1":
            pred, reason, conf = direct_judge_images_generic(
                main_img,
                sp,
                STYLE_SYSTEM_PROMPT,
                STYLE_USER_INSTRUCTION,
            )
            if pred is None and reason == RETRY_EXHAUSTED_REASON:
                return {"pair_key": pair_key, "result": None, "error": "retry_exhausted", "detail": {"style_img": sp}}
            decision = bool(pred is True and conf is not None and float(conf) > float(args.style_conf_thr))
            detail = {"status": "single", "pred": pred, "conf": conf, "reason": reason}
        else:
            decision, detail, retry_exhausted = judge_pair_voting(
                path_a=main_img,
                path_b=sp,
                system_prompt=STYLE_SYSTEM_PROMPT,
                user_instruction=STYLE_USER_INSTRUCTION,
                conf_thr=float(args.style_conf_thr),
                judge_times=int(args.style_judge_times),
                min_true=int(args.style_min_true),
            )
            if retry_exhausted:
                return {"pair_key": pair_key, "result": None, "error": "retry_exhausted", "detail": {"style_img": sp}}
            decision = bool(decision is True)
        if decision:
            passed_style += 1
        per_style_details.append({"dir": tag, "exists": True, "decision": decision, "detail": detail})
    if total_style <= 0:
        return {"pair_key": pair_key, "result": None, "error": "no_valid_style_image", "detail": {"items": per_style_details}}
    ratio = passed_style / float(total_style)
    final = 1 if ratio >= float(args.style_ratio) else 0
    return {
        "pair_key": pair_key,
        "result": final,
        "error": "",
        "detail": {
            "passed": passed_style,
            "total": total_style,
            "ratio": ratio,
            "items": per_style_details,
        },
    }


def _worker_process(model: str, base_url: str, tasks: List[Dict[str, Any]], result_queue: mp.Queue, args_obj: Any):
    global MODEL, BASE_URL
    MODEL = model
    BASE_URL = base_url
    global G_ARGS
    G_ARGS = args_obj
    for task in tasks:
        try:
            result_queue.put(_judge_one(task))
        except Exception as e:
            result_queue.put(
                {
                    "pair_key": task.get("pair_key", ""),
                    "result": None,
                    "error": f"worker_exception: {e}",
                    "detail": {},
                }
            )


def main():
    parser = argparse.ArgumentParser("按triplet key映射style id并做风格判别，输出 {pair_key:0/1} jsonl")
    parser.add_argument("--triplet-jsonl", required=True)
    parser.add_argument("--style-index-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--error-log-jsonl", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--style_conf_thr", type=float, default=0.5)
    parser.add_argument("--style_judge_times", type=int, default=3)
    parser.add_argument("--style_min_true", type=int, default=2)
    parser.add_argument("--style_ratio", type=float, default=0.66)
    parser.add_argument("--style_repeat_only_style1", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--base_url", type=str, default=BASE_URL)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--procs_per_endpoint", type=int, default=1)
    parser.add_argument("--conn_retry_times", type=int, default=5)
    parser.add_argument("--conn_retry_delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1, help="每处理多少条结果后flush一次，默认1表示逐条flush")
    args = parser.parse_args()

    style_index = read_style_index(args.style_index_jsonl)
    tasks, skipped_parse = parse_triplet_jsonl(args.triplet_jsonl, style_index)
    if not tasks:
        raise RuntimeError("没有可处理任务")

    existing_done = load_existing_done_keys(args.out_jsonl) if (not args.overwrite) else {}
    if existing_done:
        before = len(tasks)
        tasks = [t for t in tasks if t.get("pair_key") not in existing_done]
        log(f"[Resume] 已有结果 {len(existing_done)} 条，待处理 {len(tasks)}/{before}")
        if not tasks:
            log("[Resume] 全部已处理，无需继续")
            return

    rng = random.Random(args.seed)
    if args.num_samples > 0 and args.num_samples < len(tasks):
        tasks = rng.sample(tasks, args.num_samples)

    endpoints: List[Tuple[str, str]] = []
    for e in args.endpoint:
        s = str(e).strip()
        if not s:
            continue
        if "@" in s:
            m, u = s.split("@", 1)
            endpoints.append((m.strip(), u.strip()))
        else:
            endpoints.append((args.model, s))
    if not endpoints:
        endpoints = [(args.model, args.base_url)]

    pp = max(1, int(args.procs_per_endpoint))
    worker_specs: List[Tuple[str, str]] = []
    for _ in range(pp):
        for item in endpoints:
            worker_specs.append(item)
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(len(worker_specs))]
    for idx, task in enumerate(tasks):
        chunks[idx % len(worker_specs)].append(task)

    out_path = args.out_jsonl
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    err_path = args.error_log_jsonl.strip()
    if err_path:
        os.makedirs(os.path.dirname(err_path) or ".", exist_ok=True)

    result_queue: mp.Queue = mp.Queue()
    workers: List[mp.Process] = []
    for i, (m, u) in enumerate(worker_specs):
        sub = chunks[i]
        if not sub:
            continue
        p = mp.Process(target=_worker_process, args=(m, u, sub, result_queue, args))
        p.daemon = False
        p.start()
        workers.append(p)

    total = len(tasks)
    done = 0
    ok = 0
    ok_pos = 0
    errs = 0
    flush_every = max(1, int(args.flush_every))
    out_mode = "w" if args.overwrite else "a"
    if args.overwrite:
        existing_done = {}
    pbar = tqdm(total=total, desc="StyleIndexJudge", unit="pair")
    with open(out_path, out_mode, encoding="utf-8", buffering=1) as fout:
        if (not args.overwrite) and existing_done:
            pass
        ferr = open(err_path, "w", encoding="utf-8", buffering=1) if err_path else None
        try:
            while done < total:
                try:
                    rec = result_queue.get(timeout=10.0)
                    done += 1
                except queue.Empty:
                    alive = any(p.is_alive() for p in workers)
                    if alive:
                        continue
                    break
                pair_key = rec["pair_key"]
                result = rec["result"]
                if result in (0, 1):
                    fout.write(json.dumps({pair_key: int(result)}, ensure_ascii=False) + "\n")
                    ok += 1
                    if int(result) == 1:
                        ok_pos += 1
                else:
                    errs += 1
                    if ferr is not None:
                        ferr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if done % flush_every == 0:
                    fout.flush()
                    if ferr is not None:
                        ferr.flush()
                pbar.update(1)
                if done % 200 == 0 or done == total:
                    pos_ratio = (ok_pos / ok) if ok > 0 else 0.0
                    log(f"progress {done}/{total} ok={ok} err={errs} style_pass_1={ok_pos} style_pass_ratio={pos_ratio:.2%}")
        finally:
            if ferr is not None:
                ferr.close()
            pbar.close()

    for p in workers:
        p.join()

    if done < total:
        missing = total - done
        errs += missing
        log(f"[WARN] worker提前退出，未返回结果数量={missing}")
    final_ratio = (ok_pos / ok) if ok > 0 else 0.0
    log(f"DONE total={total} done={done} ok={ok} err={errs} style_pass_1={ok_pos} style_pass_ratio={final_ratio:.2%} skipped_parse={skipped_parse}")
    log(f"out_jsonl={out_path}")
    if err_path:
        log(f"error_log_jsonl={err_path}")


if __name__ == "__main__":
    main()

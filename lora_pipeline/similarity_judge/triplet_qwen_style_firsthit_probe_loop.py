#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 /data/benchmark_metrics/lora_pipeline/similarity_judge/triplet_qwen_style_firsthit_probe_loop.py \
  --image-a /data/benchmark_metrics/assets/jiegeng.png \
  --image-b /data/benchmark_metrics/assets/style.webp \
  --prompt '请判断两张图画风是否一致，只输出 0 或 1。' \
  --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.12.16:22002/v1" \
  --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.14.67:22002/v1" \
  --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.8.16:22002/v1"

"""

import argparse
import base64
import json
import math
import multiprocessing as mp
import os
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image


DEFAULT_ENDPOINT_SOURCE = "/data/benchmark_metrics/lora_pipeline/similarity_judge/triplet_qwen_style_firsthit_judge.sh"
DEFAULT_MODEL = "Qwen3-VL-30B-A3B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:22002/v1"
DEFAULT_API_KEY = "EMPTY"
RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85
Image.MAX_IMAGE_PIXELS = None

DEFAULT_SYSTEM_PROMPT = (
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

DEFAULT_USER_PROMPT = (
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def trim_text(text: str, limit: int = 300) -> str:
    s = (text or "").replace("\n", " ").replace("\r", " ")
    return s[:limit]


def split_endpoint_spec(spec: str, default_model: str) -> Tuple[str, str]:
    s = str(spec).strip()
    if not s:
        raise ValueError("empty endpoint spec")
    if "@" in s:
        model, url = s.split("@", 1)
        model = model.strip()
        url = url.strip()
        if not model or not url:
            raise ValueError(f"invalid endpoint spec: {spec}")
        return model, url
    return default_model, s


def load_endpoints_from_source(path: str, default_model: str) -> List[Tuple[str, str]]:
    if not path or not os.path.isfile(path):
        return []
    pattern = re.compile(r'^\s*endpoint\d+\s*=\s*(["\'])(.+?)\1\s*$')
    endpoints: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            model, url = split_endpoint_spec(m.group(2), default_model)
            endpoints.append((model, url))
    return endpoints


def resolve_endpoints(
    endpoint_args: Sequence[str],
    endpoint_source: str,
    default_model: str,
    default_base_url: str,
) -> List[Tuple[str, str]]:
    endpoints: List[Tuple[str, str]] = []
    for item in endpoint_args:
        s = str(item).strip()
        if not s:
            continue
        endpoints.append(split_endpoint_spec(s, default_model))
    if endpoints:
        return endpoints
    endpoints = load_endpoints_from_source(endpoint_source, default_model)
    if endpoints:
        return endpoints
    return [(default_model, default_base_url)]


def read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def load_rgb_image(path: str) -> Image.Image:
    img = Image.open(BytesIO(read_image_bytes(path)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_keep_long_side(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    side = max(width, height)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def make_image_data_uri(path: str) -> str:
    img = load_rgb_image(path)
    img = resize_keep_long_side(img, RESIZE_MAX_SIDE)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return s


def extract_text_from_choice(choice: Dict[str, Any]) -> str:
    message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def collect_top_logprobs_mapping(resp_json: Dict[str, Any]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    choices = resp_json.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return mapping
    choice0 = choices[0]
    logprobs = choice0.get("logprobs", None)
    if logprobs is None:
        message = choice0.get("message", {}) if isinstance(choice0.get("message"), dict) else {}
        logprobs = message.get("logprobs", None)
    if not isinstance(logprobs, dict):
        return mapping
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        first = content[0] if isinstance(content[0], dict) else None
        if isinstance(first, dict):
            top = first.get("top_logprobs")
            if isinstance(top, list):
                for item in top:
                    if not isinstance(item, dict):
                        continue
                    token = item.get("token")
                    logprob = item.get("logprob")
                    if isinstance(token, str) and isinstance(logprob, (int, float)):
                        mapping[token] = float(logprob)
    return mapping


def extract_01_logprobs(resp_json: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    mapping_raw = collect_top_logprobs_mapping(resp_json)
    logp0 = None
    logp1 = None
    for token, logprob in mapping_raw.items():
        stripped = token.strip()
        if stripped == "0":
            logp0 = logprob
        elif stripped == "1":
            logp1 = logprob
    return logp0, logp1, mapping_raw


def safe_exp(x: float) -> float:
    if x > 60:
        return math.exp(60)
    if x < -60:
        return math.exp(-60)
    return math.exp(x)


def softmax2(logp0: float, logp1: float) -> Tuple[float, float]:
    m = max(logp0, logp1)
    a0 = safe_exp(logp0 - m)
    a1 = safe_exp(logp1 - m)
    denom = a0 + a1
    if denom <= 0:
        return 0.5, 0.5
    return a0 / denom, a1 / denom


def build_messages(
    image_a_data_uri: str,
    image_b_data_uri: str,
    system_prompt: str,
    prompt: str,
    output_instruction: str,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "下面是图片 A："},
        {"type": "image_url", "image_url": {"url": image_a_data_uri}},
        {"type": "text", "text": "下面是图片 B："},
        {"type": "image_url", "image_url": {"url": image_b_data_uri}},
    ]
    if output_instruction.strip():
        user_content.append({"type": "text", "text": output_instruction})
    messages.append({"role": "user", "content": user_content})
    return messages


def call_qwen_chat_with_retry(
    model: str,
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    connect_timeout_sec: float,
    request_timeout_sec: float,
    retry_times: int,
    retry_delay: float,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    total_attempts = max(0, int(retry_times)) + 1
    last_result: Dict[str, Any] = {
        "ok": False,
        "model": model,
        "url": url,
        "http_code": None,
        "elapsed_sec": None,
        "error": "unknown_error",
        "body": "",
        "resp_json": None,
    }

    for attempt in range(1, total_attempts + 1):
        started = time.time()
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(float(connect_timeout_sec), float(request_timeout_sec)),
            )
            elapsed = time.time() - started
            body = resp.text
            if 200 <= int(resp.status_code) < 300:
                try:
                    return {
                        "ok": True,
                        "model": model,
                        "url": url,
                        "http_code": int(resp.status_code),
                        "elapsed_sec": elapsed,
                        "error": "",
                        "body": "",
                        "resp_json": resp.json(),
                    }
                except Exception as e:
                    last_result = {
                        "ok": False,
                        "model": model,
                        "url": url,
                        "http_code": int(resp.status_code),
                        "elapsed_sec": elapsed,
                        "error": f"json_decode_error: {e}",
                        "body": trim_text(body),
                        "resp_json": None,
                    }
            else:
                last_result = {
                    "ok": False,
                    "model": model,
                    "url": url,
                    "http_code": int(resp.status_code),
                    "elapsed_sec": elapsed,
                    "error": f"http_{resp.status_code}",
                    "body": trim_text(body),
                    "resp_json": None,
                }
        except Exception as e:
            elapsed = time.time() - started
            last_result = {
                "ok": False,
                "model": model,
                "url": url,
                "http_code": None,
                "elapsed_sec": elapsed,
                "error": str(e),
                "body": "",
                "resp_json": None,
            }
        if attempt < total_attempts:
            time.sleep(float(retry_delay))
    return last_result


def parse_response(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp_json.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return {
            "status": "bad_resp",
            "text": "",
            "pred_char": "",
            "conf": None,
            "detail": "返回结构异常(无 choices)",
        }

    text = strip_code_fences(extract_text_from_choice(choices[0])).strip()
    pred_char = ""
    for ch in text:
        if not ch.isspace():
            pred_char = ch
            break

    logp0, logp1, mapping_raw = extract_01_logprobs(resp_json)
    conf = None
    if pred_char == "0" and logp0 is not None and logp1 is not None:
        p0, _ = softmax2(logp0, logp1)
        conf = p0
    elif pred_char == "1" and logp0 is not None and logp1 is not None:
        _, p1 = softmax2(logp0, logp1)
        conf = p1

    if pred_char in ("0", "1"):
        detail = f"pred={pred_char}"
        if conf is not None:
            detail += f", conf={conf:.3f}"
        return {
            "status": "ok",
            "text": text,
            "pred_char": pred_char,
            "conf": conf,
            "detail": detail,
        }

    return {
        "status": "bad_resp",
        "text": text,
        "pred_char": pred_char,
        "conf": conf,
        "detail": f"输出不是 0/1 (got={text!r}, top_logprobs_keys={list(mapping_raw.keys())[:8]})",
    }


def worker_loop(
    worker_id: int,
    model: str,
    base_url: str,
    image_a_data_uri: str,
    image_b_data_uri: str,
    args: argparse.Namespace,
) -> None:
    if float(args.startup_stagger_sec) > 0:
        time.sleep(worker_id * float(args.startup_stagger_sec))

    messages = build_messages(
        image_a_data_uri=image_a_data_uri,
        image_b_data_uri=image_b_data_uri,
        system_prompt=args.system_prompt,
        prompt=args.prompt,
        output_instruction=args.output_instruction,
    )

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": float(args.temperature),
        "messages": messages,
        "max_tokens": int(args.max_tokens),
    }
    if args.logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(args.top_logprobs)
        payload["top_k"] = int(args.top_logprobs)

    url = base_url.rstrip("/") + "/chat/completions"
    log(f"[START] worker={worker_id} model={model} url={url} interval={args.interval_sec}s")

    seq = 0
    while True:
        seq += 1
        call_result = call_qwen_chat_with_retry(
            model=model,
            base_url=base_url,
            api_key=args.api_key,
            payload=payload,
            connect_timeout_sec=float(args.connect_timeout_sec),
            request_timeout_sec=float(args.request_timeout_sec),
            retry_times=int(args.conn_retry_times),
            retry_delay=float(args.conn_retry_delay),
        )

        if not call_result["ok"]:
            elapsed = call_result["elapsed_sec"]
            elapsed_str = f"{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else "NA"
            log(
                "[FAIL] "
                f"worker={worker_id} model={model} url={base_url} seq={seq} "
                f"code={call_result['http_code']} time={elapsed_str} "
                f"err={trim_text(str(call_result['error']))} body={call_result['body']}"
            )
        else:
            parsed = parse_response(call_result["resp_json"])
            elapsed = call_result["elapsed_sec"]
            elapsed_str = f"{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else "NA"
            if parsed["status"] == "ok":
                conf_str = f"{parsed['conf']:.3f}" if isinstance(parsed["conf"], (int, float)) else "NA"
                log(
                    "[OK] "
                    f"worker={worker_id} model={model} url={base_url} seq={seq} "
                    f"code={call_result['http_code']} time={elapsed_str} "
                    f"pred={parsed['pred_char']} conf={conf_str} text={trim_text(parsed['text'])}"
                )
            else:
                log(
                    "[BAD_RESP] "
                    f"worker={worker_id} model={model} url={base_url} seq={seq} "
                    f"code={call_result['http_code']} time={elapsed_str} "
                    f"detail={trim_text(parsed['detail'])}"
                )

        if int(args.max_loops) > 0 and seq >= int(args.max_loops):
            break
        time.sleep(float(args.interval_sec))


def validate_file(path: str, label: str) -> None:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser("真实两图判别探活：每个 endpoint 一个进程，循环请求，不落盘")
    parser.add_argument("--image-a", required=True, help="图片 A 路径")
    parser.add_argument("--image-b", required=True, help="图片 B 路径")
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT, help="判别 prompt，会作为 user content 的第一段文本")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="可选 system prompt")
    parser.add_argument("--output-instruction", default="只输出一个字符：0 或 1。", help="附加在 user content 末尾的输出约束")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--endpoint", action="append", default=[], help="可重复传入，格式: model@http://host:port/v1")
    parser.add_argument("--endpoint-source", default=DEFAULT_ENDPOINT_SOURCE, help="若未显式传 --endpoint，则从该 shell 脚本解析 endpointN=...")
    parser.add_argument("--interval-sec", type=float, default=2.0, help="每轮请求间隔秒数")
    parser.add_argument("--startup-stagger-sec", type=float, default=0.2, help="worker 启动错峰秒数")
    parser.add_argument("--max-loops", type=int, default=0, help="每个 worker 最多循环多少次；0 表示无限循环")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.set_defaults(logprobs=True)
    parser.add_argument("--logprobs", action="store_true", dest="logprobs", help="开启 logprobs 请求")
    parser.add_argument("--no-logprobs", action="store_false", dest="logprobs", help="关闭 logprobs 请求")
    parser.add_argument("--top-logprobs", type=int, default=8)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--connect-timeout-sec", type=float, default=3.0)
    parser.add_argument("--request-timeout-sec", type=float, default=60.0)
    parser.add_argument("--conn-retry-times", type=int, default=5, help="单轮请求内部重试次数，默认与现有 judge 脚本一致")
    parser.add_argument("--conn-retry-delay", type=float, default=2.0)
    args = parser.parse_args()

    validate_file(args.image_a, "image-a")
    validate_file(args.image_b, "image-b")

    endpoints = resolve_endpoints(
        endpoint_args=args.endpoint,
        endpoint_source=args.endpoint_source,
        default_model=args.model,
        default_base_url=args.base_url,
    )
    if not endpoints:
        raise RuntimeError("no endpoints found")

    image_a_data_uri = make_image_data_uri(args.image_a)
    image_b_data_uri = make_image_data_uri(args.image_b)

    log(f"[Host] image_a={args.image_a}")
    log(f"[Host] image_b={args.image_b}")
    log(f"[Host] endpoints={len(endpoints)}")
    for idx, (model, url) in enumerate(endpoints, start=1):
        log(f"[Host] endpoint[{idx}] model={model} url={url}")

    workers: List[mp.Process] = []
    try:
        for idx, (model, url) in enumerate(endpoints, start=1):
            p = mp.Process(
                target=worker_loop,
                args=(idx, model, url, image_a_data_uri, image_b_data_uri, args),
            )
            p.daemon = False
            p.start()
            workers.append(p)

        for p in workers:
            p.join()
    except KeyboardInterrupt:
        log("[Host] interrupted, terminating workers")
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=1.0)


if __name__ == "__main__":
    main()

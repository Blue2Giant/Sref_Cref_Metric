#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多内容参考图：共同内容提取 + 二分类(0/1) + logprobs置信度

用法示例：
python qwen_common_content_binary.py \
  --refs ref1.png ref2.png ref3.png \
  --stylized gen.png \
  --base-url http://10.191.0.41:22002/v1 \
  --model Qwen3-VL-30B-A3B-Instruct
"""

import os
import re
import json
import math
import argparse
import base64
import mimetypes
from typing import Optional, Tuple, List

from openai import OpenAI

# ========== Prompt：共同内容 + 二分类 ==========
COMMON_CONTENT_BINARY_PROMPT = """
你将看到若干参考图（R1, R2, ...），它们大体应当共享同一个主体/主题（例如同一个人、同一件衣物、同一类物体或同一场景），以及一张待判定图（G）。

任务分两步：
1) 从参考图集合中总结“最稳定、最一致”的共同内容/主题，用一句中文描述。优先考虑：
   - 人物：身份一致性/性别年龄段、发型发色、衣物类别与主色、配饰（眼镜/帽子/首饰等）
   - 物体：类别、主色、材质、数量（显著物体）
   - 场景：室内/室外、地点类型、关键背景元素
   如果参考图存在少量不一致，忽略明显离群图，以多数稳定特征为准。
   如果确实无法总结出明确共同内容，写“无明显共同主题”。

2) 判断 G 是否包含该共同内容/主题：
   - 若 G 与参考集合共享该共同内容/主题（允许风格、光照、渲染差异），输出 LABEL=1
   - 若不共享或明显矛盾，输出 LABEL=0

输出格式必须严格只包含两行（不要输出其它任何文字/解释/多余空行）：
COMMON: <一句话中文描述，不要换行>
LABEL: <0 或 1>
""".strip()

# 解析两行输出
RE_COMMON = re.compile(r"^\s*COMMON\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
RE_LABEL = re.compile(r"^\s*LABEL\s*:\s*([01])\s*$", re.IGNORECASE | re.MULTILINE)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ========== 工具函数 ==========
def path_to_data_url(path: str) -> str:
    """把本地图片转成 data URL，给 Qwen-VL 用。"""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_messages(ref_paths: List[str], stylized_path: str):
    """
    构造 Chat Completions messages：
    - R1..Rn：参考图
    - G：待判定图
    - 最后附加文字 prompt
    """
    content = []

    for i, p in enumerate(ref_paths, 1):
        content.append({"type": "text", "text": f"Reference R{i}:"})
        content.append(
            {"type": "image_url", "image_url": {"url": path_to_data_url(p)}}
        )

    content.append({"type": "text", "text": "Target G:"})
    content.append(
        {"type": "image_url", "image_url": {"url": path_to_data_url(stylized_path)}}
    )

    content.append({"type": "text", "text": COMMON_CONTENT_BINARY_PROMPT})
    return [{"role": "user", "content": content}]


def parse_common_and_label(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    解析 COMMON 与 LABEL。
    先按两行格式解析；若失败，尝试 JSON 兜底。
    """
    m1 = RE_COMMON.search(text)
    m2 = RE_LABEL.search(text)
    if m1 and m2:
        common = m1.group(1).strip()
        label = int(m2.group(1))
        return common, label

    # JSON 兜底：允许模型偶发输出 {"common": "...", "label": 1} 之类
    try:
        obj = json.loads(text)
        common = obj.get("common") or obj.get("common_content") or obj.get("COMMON")
        label = obj.get("label") or obj.get("LABEL")
        if common is not None:
            common = str(common).strip()
        if label is not None:
            label = int(label)
            if label not in (0, 1):
                label = None
        return common, label
    except Exception:
        return None, None


def _norm_digit(tok: str) -> Optional[str]:
    """
    把 token 归一化为 '0' 或 '1'（尽量容忍标点/空白）。
    注意：如果 token 是 '1,' 或 '1\\n' 也会被识别。
    """
    s = tok.strip()
    # 去掉常见包裹符号
    s = s.strip(" \t\r\n,:;}]>)\"'")
    if s in ("0", "1"):
        return s
    return None


def extract_label_confidence(resp, label: Optional[int]) -> Optional[float]:
    """
    从 logprobs 估算二分类标签的置信度。

    优先策略（更“二分类”）：
      - 找到 LABEL 行附近的第一个 0/1 token
      - 若 top_logprobs 同时出现 0 和 1，则对 {0,1} 做 softmax 得到 P(label)
    兜底策略：
      - 若只拿到当前 label token 的 logprob（没有另一类），返回 exp(logprob) 作为“该token概率”近似
    """
    if label is None:
        return None

    choice = resp.choices[0]
    lp = getattr(choice, "logprobs", None)
    if not lp or not getattr(lp, "content", None):
        return None

    items = lp.content

    # 先重建输出文本的“token拼接流”，定位到 'LABEL' 之后再找数字
    acc = ""
    label_zone = False

    for idx, item in enumerate(items):
        tok = item.token
        acc += tok

        if (not label_zone) and ("LABEL" in acc.upper()):
            label_zone = True

        if not label_zone:
            continue

        d = _norm_digit(tok)
        if d is None:
            continue

        # 命中到 0/1 token（尽量认为这是 label 位）
        digit_set = {"0", "1"}
        target = str(label)

        # 收集候选 logprob：来自 top_logprobs（更适合二分类 softmax）
        cand_logs = {}
        tops = item.top_logprobs or []
        for t in tops:
            td = _norm_digit(t.token)
            if td in digit_set:
                cand_logs[td] = t.logprob

        # 把当前token也算进去（有时 top_logprobs 不含自身）
        if d in digit_set and d not in cand_logs and hasattr(item, "logprob"):
            cand_logs[d] = item.logprob

        if len(cand_logs) >= 2 and ("0" in cand_logs) and ("1" in cand_logs):
            # 对 {0,1} 做 softmax
            max_log = max(cand_logs["0"], cand_logs["1"])
            p0 = math.exp(cand_logs["0"] - max_log)
            p1 = math.exp(cand_logs["1"] - max_log)
            s = p0 + p1
            if s <= 0:
                return None
            probs = {"0": p0 / s, "1": p1 / s}
            return float(probs.get(target, None))

        # 兜底：返回该 token 的概率 exp(logprob)
        if hasattr(item, "logprob") and item.logprob is not None:
            try:
                p = math.exp(item.logprob)
                # 防止极端数值
                if p < 0.0:
                    p = 0.0
                if p > 1.0:
                    p = 1.0
                return float(p)
            except OverflowError:
                return None

        return None

    return None


# ========== 核心：调用一次 Qwen3-VL ==========
def run_common_content_binary(
    client: OpenAI,
    model: str,
    ref_paths: List[str],
    stylized_path: str,
    max_tokens: int = 128,
    top_logprobs: int = 20,
) -> Tuple[Optional[str], Optional[int], Optional[float], str]:
    """
    返回：
      common: 共同内容一句话
      label: 0/1
      confidence: 基于 logprobs 的置信度（0~1），可能为 None
      raw: 原始输出
    """
    if not ref_paths:
        raise ValueError("ref_paths 不能为空，至少需要一张参考图")

    messages = build_messages(ref_paths, stylized_path)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    raw = (resp.choices[0].message.content or "").strip()
    common, label = parse_common_and_label(raw)
    conf = extract_label_confidence(resp, label)
    return common, label, conf, raw


# ========== CLI 入口 ==========
def _validate_image_paths(paths: List[str]) -> List[str]:
    ok = []
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到文件: {p}")
        ext = os.path.splitext(p)[1].lower()
        if ext not in IMG_EXTS:
            # 不强制，但给个提醒也行；这里直接放行（有些是 .jfif 或者无后缀）
            pass
        ok.append(p)
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 多参考图：共同内容提取 + 0/1 判定 + logprobs 置信度"
    )
    parser.add_argument(
        "--refs",
        nargs="+",
        required=True,
        help="多张内容参考图路径（同一主体/主题）",
    )
    parser.add_argument(
        "--stylized",
        required=True,
        help="待判定图 (G)",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI 兼容 base_url，如 http://host:port/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API Key（默认读环境变量 OPENAI_API_KEY）",
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
        "--top-logprobs",
        type=int,
        default=20,
        help="logprobs候选数量（越大越可能同时包含 0 和 1）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
    )
    args = parser.parse_args()

    ref_paths = _validate_image_paths(args.refs)
    stylized_path = _validate_image_paths([args.stylized])[0]

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    common, label, conf, raw = run_common_content_binary(
        client=client,
        model=args.model,
        ref_paths=ref_paths,
        stylized_path=stylized_path,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
    )

    print("=== Raw output ===")
    print(raw)
    print("=== Parsed COMMON ===")
    print(common)
    print("=== Parsed LABEL (0/1) ===")
    print(label)
    print("=== Confidence (from logprobs) ===")
    print(conf)


if __name__ == "__main__":
    main()

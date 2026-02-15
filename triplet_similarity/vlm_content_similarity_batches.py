#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多进程（简化版）：server 推理，不做 GPU 绑定

遍历 root 下所有 model_dir：
- refs：demo_images（或 content_100）
- eval：eval_images
流程：
1) 用 refs 提取共同主题 COMMON（refs 过多则采样）
2) 对 eval_images 每张图：
   - 输出 LABEL=0/1
   - 从 logprobs 估计置信度 conf
   - similarity = LABEL * conf
3) 写入 JSON（overall_mean_similarity 是 eval 的平均 similarity）

并行逻辑：
- --num-workers 指定进程数
- 将 model_dirs 均匀分配给各进程

关键增强：
- 加入 system prompt（强约束：只输出要求格式、不解释）
- COMMON 与 LABEL 的 prompt 变得更细、更严格：
  * 一旦锁定共同主题，强制忽略画风/光照/背景/构图等差异，只看主体/主题一致性
  * 严格提高相似度门槛：证据不足宁可判 0
  * 人脸身份核验更苛刻：面部不可可靠比对 => 默认 0
- LABEL 输出强制为“单个数字 0/1 的一行”，提升 logprobs 提取稳定性
"""

import os
import re
import io
import json
import math
import base64
import argparse
import mimetypes
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional

from megfile.smart import (
    smart_open as mopen,
    smart_listdir,
    smart_exists,
    smart_scandir,
    smart_makedirs,
    smart_isdir,
)
from openai import OpenAI


# ==================== System Prompt（强约束） ====================

SYSTEM_PROMPT = """
你是一名“严格的图像内容一致性判别器”，目标是用极其严格且一致的规则做二分类。
必须遵守以下硬规则（任何时候都不能违反）：

[输出约束]
- 你只能按用户指令输出指定的格式，绝对不要输出解释、推理过程、理由、补充说明、标点或多余空行。
- 若指令要求“一行输出”，你必须严格只输出一行。
- 当被要求输出 0/1 时，你必须只输出单个字符：0 或 1（同一行、无其它字符）。

[判别理念]
- 只比较“内容/主体/主题”一致性，严格忽略风格类因素：画风/渲染方式、滤镜、光照、色彩分级、清晰度、噪声、分辨率、构图、背景变化、姿态动作、拍摄角度与裁剪。
- 一旦你识别到“共同主题”的核心要素，就把其它差异当作噪声；但同时提高严格度：只有在强证据支持高度一致时才判 1。
- 证据不足或不可核验时，宁可判 0（偏保守，减少误判 1）。

[人脸/人物身份特别严格]
- 若主题涉及“同一个人/同一身份”，必须优先用脸部身份特征核验（五官结构组合、脸型轮廓、关键特征点关系）。
- 发型、妆容、衣服、背景、画风相似都不能作为同一人的充分证据。
- 若面部不清晰/遮挡/角度极端/分辨率不足导致无法可靠比对：默认输出 0（除非存在几乎唯一且强的身份线索）。
""".strip()


# ==================== Prompts（超详细/严格） ====================
# ==================== Prompts（更详细 / 更清晰 / 更严格版；去掉“从强到弱优先级”） ====================

COMMON_ONLY_PROMPT = """
你将看到若干参考图（R1, R2, ...）。它们大体应当共享同一个“共同内容/主题”（例如同一个人、同一件物体/产品、同一种生物、同一个地点/场景、同一个品牌/标志性对象等）。

你的任务：从参考图集合中提炼“最稳定、最一致、最能唯一指向同一共同主题”的内容，用一句中文概括。
你必须只总结“内容/主体/主题”，不要总结风格因素。

【你必须忽略的差异（永远不要写入 COMMON）】
- 画风/渲染方式（写实/二次元/插画/3D/照片感/油画/水彩等）
- 光照、曝光、色彩分级、滤镜、噪声颗粒、清晰度、分辨率
- 构图、视角、裁剪范围、主体占比大小、姿态动作、表情变化
- 背景变化与环境杂物（除非你确定背景本身就是共同主题的核心组成）

【你必须坚持的原则】
1) 多数一致原则：如果少数参考图明显离群（主题不同/主体不同），将其视为噪声忽略，以“多数图稳定出现的共同主题”作为最终 COMMON。
2) 高判别力原则：COMMON 要尽量用“高辨识度特征”表达主题，避免过于宽泛的描述（例如“一个人”“一双鞋”“一只动物”过于泛）。
3) 稳定性原则：只写“跨大多数参考图都稳定出现”的内容；不稳定、偶发、可变的细节不要写。
4) 单主题原则：COMMON 只能描述一个共同主题，不要同时描述两个不同主题；如果确实无法收敛为单一主题，输出“无明显共同主题”。

【你可能遇到的共同主题类别（示例，不是优先级）】
- 人物身份：同一个人（同一身份/同一角色）
- 物体/产品：同一种具体物体（鞋/包/手表/车/飞机/家具/手机等），或同一款式/同一品牌特征
- 动物/生物：同一种动物且外观特征稳定（品种/毛色/体型特征等）
- 场景/地点：同一地点类型或同一具体场景（海滩/办公室/卧室/街道/教室等）且关键元素组合稳定
- 标志性符号：稳定出现的logo/图案/文字/标识/徽章（若清晰可辨）
- 组合主题：主体 + 强绑定属性（例如“同一人 + 标志性眼镜 + 黑色夹克”，或“同款手表 + 圆表盘 + 金属表带”）
  注意：组合主题仅在这些属性在多数参考图中都稳定出现时才允许写入。

【当主题可能是“同一个人”时（非常关键）】
- 你需要寻找能支持“同一身份”的稳定线索：脸型轮廓、五官结构组合、发型发色（辅助）、标志性配饰（辅助）、稳定的穿着类型与主色块（仅辅助）。
- 不允许用“像某明星/某具体真实姓名”来描述；只描述可观察特征。
- COMMON 里不要写“画风像xxx”，也不要写“照片/动漫风”。

【当主题可能是“物体/产品/物件”时】
- COMMON 应包含：明确类别 + 关键结构/部件组合（例如“圆形表盘+金属表带”）+ 稳定主色/材质（若稳定可见）。
- 如果类别相同但结构差异很大，应当认为无法收敛，倾向输出“无明显共同主题”。

【当主题可能是“场景/地点”时】
- COMMON 应包含：场景类型 + 稳定出现的关键元素组合（例如“海滩+海浪+沙滩”）。
- 不要被光照、色调、天气变化误导；这些不写进 COMMON。

【无法总结时】
- 如果参考图主题高度分散，或排除离群后仍无法得到稳定共同主题，输出：
  无明显共同主题

输出格式必须严格只包含一行（不要输出其它任何文字/解释/多余空行）：
COMMON: <一句话中文描述，不要换行>
""".strip()


LABEL_ONLY_PROMPT_TEMPLATE = """
已知参考图集合的共同内容/主题为：
COMMON: {common}

现在你将看到一张目标图（G）。你的任务是：以“非常严格”的标准判断 G 是否包含并高度匹配上述共同主题。
你必须只输出 0 或 1（单个字符，一行），不要输出任何解释或其它字符。

【必须忽略的差异（不允许影响判定）】
- 画风/渲染方式、滤镜、光照、色彩分级、噪声、清晰度、分辨率
- 构图、视角、裁剪、主体大小、姿态动作、表情变化
- 背景变化、环境杂物、摄影/绘画技法差异

【判定核心：只看“共同主题是否存在且关键特征高度一致”】【更严格】
你需要做到“仔细核对关键证据”，而不是凭大概相似判断：
- 只有当你能找到足够强的证据支持“同一共同主题”时才输出 1
- 只要存在关键冲突、或证据不足无法确认，就输出 0（偏保守，宁可漏判，不可误判）

【输出 1 的必要条件（必须同时满足）】
1) G 中确实出现 COMMON 所描述的“核心要素”（主体/物体/场景关键组合）
2) 关键特征与 COMMON 高度一致，没有明显矛盾
3) 证据充分：你几乎可以确信是同一共同主题（而不是“看起来差不多”）

【输出 0 的条件（任一满足即可）】
a) G 中根本没有 COMMON 的核心要素
b) 有相似元素但关键属性冲突（类别不同、结构不同、身份不同、关键部件不同）
c) 证据不足无法核验（遮挡严重、主体太小、过度模糊、只出现局部且无法确认）
d) 只是风格/色调/构图相似，但内容主题不一致

【当 COMMON 是“人物身份/同一个人”时（极其严格，必须逐项核验）】
- 你必须优先用“脸部身份特征”判断：五官结构组合、脸型轮廓、关键特征点相对关系。
- 仅凭发型、妆容、衣服、身材、背景、画风相似，不能输出 1。
- 若面部不可可靠比对（太小/模糊/遮挡/角度极端/低像素）：默认输出 0。
- 若出现明显身份冲突（脸型/五官结构明显不同）：输出 0。

【当 COMMON 是“物体/产品/物件”时（严格核对结构）】
- 类别必须一致（例如鞋=鞋、手表=手表、飞机=飞机）。
- 关键结构/部件组合必须一致；仅颜色接近但结构不同 => 0。
- 仅出现局部且不足以确认同一物体/同一款式 => 0。

【当 COMMON 是“动物/生物”时（严格核对物种与外观特征）】
- 物种/类别必须一致；明显不同物种 => 0。
- 若仅“毛色相近”但体态/关键特征不同 => 0。
- 主体太小或不可辨 => 0。

【当 COMMON 是“场景/地点”时（严格核对关键元素组合）】
- 必须匹配“场景类型 + 关键元素组合”；仅色调/构图相似不算。
- 若关键元素缺失或冲突 => 0。

【当 COMMON 是“标志性符号/Logo/文字”时（严格核对可辨识度）】
- 必须能在 G 中清晰辨认出该符号/文字/标识，且与 COMMON 一致。
- 如果模糊到无法确认 => 0。
- 仅出现相似图形但细节不一致 => 0。

【最终输出格式（极重要）】
- 只输出单个字符：0 或 1
- 同一行，不要加“LABEL:”，不要空格，不要标点，不要其它文字

输出：
<0 或 1>
""".strip()


RE_COMMON = re.compile(r"^\s*COMMON\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# 兼容两种：纯 "0/1" 或 "LABEL: 0/1"（以防少数模型不听话）
RE_LABEL = re.compile(r"^\s*(?:LABEL\s*:\s*)?([01])\s*$", re.IGNORECASE | re.MULTILINE)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ==================== megfile helpers ====================

def is_remote_path_megfile(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("oss://")


def join_path(root: str, name: str) -> str:
    return root.rstrip("/") + "/" + name.lstrip("/")


def iter_model_dirs_megfile(root: str) -> List[str]:
    """
    枚举 root 下的所有一级子目录（model_id 目录）。
    支持本地和桶。
    """
    dirs: List[str] = []
    if is_remote_path_megfile(root):
        try:
            for entry in smart_scandir(root):
                try:
                    if entry.is_dir():
                        dirs.append(entry.path)
                except Exception:
                    continue
        except FileNotFoundError:
            return []
    else:
        path_root = Path(root)
        if not path_root.is_dir():
            return []
        for p in path_root.iterdir():
            if p.is_dir():
                dirs.append(str(p))

    dirs.sort()
    return dirs


def dir_exists_megfile(path: str) -> bool:
    """判断目录是否存在（本地 / 桶）。"""
    if is_remote_path_megfile(path):
        return smart_exists(path) and smart_isdir(path)
    return os.path.isdir(path)


def list_images_recursive_megfile(root: str) -> List[str]:
    """
    递归列出 root 下所有图片文件。
    支持本地路径和桶路径。
    """
    paths: List[str] = []
    if is_remote_path_megfile(root):
        stack = [root.rstrip("/")]
        while stack:
            cur = stack.pop()
            try:
                for entry in smart_scandir(cur):
                    try:
                        if entry.is_dir():
                            stack.append(entry.path)
                        else:
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in IMG_EXTS:
                                paths.append(entry.path)
                    except Exception:
                        continue
            except FileNotFoundError:
                continue
    else:
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        for p in root_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))

    paths.sort()
    return paths


# ==================== sampling / chunks ====================

def split_into_chunks(lst: List[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [lst]
    total = len(lst)
    if total == 0:
        return []
    n = min(n, total)
    base, extra = divmod(total, n)
    chunks: List[List[str]] = []
    start = 0
    for i in range(n):
        length = base + (1 if i < extra else 0)
        end = start + length
        chunks.append(lst[start:end])
        start = end
    return chunks


def select_evenly_spaced(items: List[str], k: int) -> List[str]:
    """均匀抽样 k 个，保证覆盖面"""
    if k <= 0 or len(items) <= k:
        return items
    if k == 1:
        return [items[len(items) // 2]]
    n = len(items)
    idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    out, seen = [], set()
    for ix in idxs:
        if ix not in seen:
            out.append(items[ix])
            seen.add(ix)
    return out


# ==================== image encode ====================

def smart_read_bytes(path: str) -> bytes:
    with mopen(path, "rb") as f:
        return f.read()


def encode_image_to_data_url(path: str, max_side: int = 1024, jpeg_quality: int = 90) -> str:
    """
    megfile 路径图片 -> data URL
    默认 resize + 转 JPEG 以降低请求体积（更适合大批量）
    """
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    raw = smart_read_bytes(path)

    if max_side and max_side > 0:
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            img.load()

            w, h = img.size
            m = max(w, h)
            if m > max_side:
                scale = max_side / float(m)
                nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                img = img.resize((nw, nh), Image.LANCZOS)

            if img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            raw = buf.getvalue()
            mime = "image/jpeg"
        except Exception:
            pass

    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ==================== message builders（加入 system prompt） ====================

def build_messages_for_common(ref_paths: List[str], img_max_side: int, jpeg_quality: int):
    content = []
    for i, p in enumerate(ref_paths, 1):
        content.append({"type": "text", "text": f"Reference R{i}:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(p, img_max_side, jpeg_quality)}})
    content.append({"type": "text", "text": COMMON_ONLY_PROMPT})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_messages_for_label(common: str, g_path: str, img_max_side: int, jpeg_quality: int):
    prompt = LABEL_ONLY_PROMPT_TEMPLATE.format(common=common)
    content = [
        {"type": "text", "text": "Target G:"},
        {"type": "image_url", "image_url": {"url": encode_image_to_data_url(g_path, img_max_side, jpeg_quality)}},
        {"type": "text", "text": prompt},
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ==================== parsing ====================

def parse_common(text: str) -> Optional[str]:
    m = RE_COMMON.search(text or "")
    return m.group(1).strip() if m else None


def parse_label(text: str) -> Optional[int]:
    m = RE_LABEL.search(text or "")
    return int(m.group(1)) if m else None


# ==================== logprobs confidence ====================

def _norm_digit(tok: str) -> Optional[str]:
    s = (tok or "").strip()
    # 去掉常见标点/括号/引号，但不去掉字母（避免把 "LABEL:1" 误判为纯数字）
    s = s.strip(" \t\r\n,:;}]>)\"'")
    return s if s in ("0", "1") else None


def extract_label_confidence(resp, label: Optional[int]) -> Optional[float]:
    """
    估计 LABEL 的置信度：
    - 找到输出中的第一个 0/1 token
    - 若 top_logprobs 同时有 0 和 1 -> softmax 得到 P(label)
    - 否则退化为 exp(logprob)
    """
    if label is None:
        return None
    choice = resp.choices[0]
    lp = getattr(choice, "logprobs", None)
    if not lp or not getattr(lp, "content", None):
        return None

    target = str(label)

    for item in lp.content:
        d = _norm_digit(getattr(item, "token", ""))
        if d is None:
            continue

        digit_logs: Dict[str, float] = {}
        tops = getattr(item, "top_logprobs", None) or []
        for t in tops:
            td = _norm_digit(getattr(t, "token", ""))
            if td in ("0", "1"):
                digit_logs[td] = t.logprob

        if d not in digit_logs and getattr(item, "logprob", None) is not None:
            digit_logs[d] = item.logprob

        if ("0" in digit_logs) and ("1" in digit_logs):
            max_log = max(digit_logs["0"], digit_logs["1"])
            p0 = math.exp(digit_logs["0"] - max_log)
            p1 = math.exp(digit_logs["1"] - max_log)
            s = p0 + p1
            if s <= 0:
                return None
            probs = {"0": p0 / s, "1": p1 / s}
            return float(probs.get(target, None))

        if getattr(item, "logprob", None) is not None:
            try:
                p = math.exp(item.logprob)
                return float(min(1.0, max(0.0, p)))
            except OverflowError:
                return None

        return None

    return None


# ==================== per-model_dir 处理 ====================

def process_single_model_dir(
    model_dir: str,
    client: OpenAI,
    model_name: str,
    output_name: str,
    overwrite: bool,
    probe_mode: str,
    content_dir_name: str,
    max_refs: int,
    max_eval: int,
    img_max_side: int,
    jpeg_quality: int,
    common_max_tokens: int,
    label_max_tokens: int,
    top_logprobs: int,
    save_raw: bool,
) -> None:
    demo_dir = join_path(model_dir, "demo_images")
    eval_dir = join_path(model_dir, "eval_images")
    content_dir = join_path(model_dir, content_dir_name)
    out_json = join_path(model_dir, output_name)

    probe_dir = demo_dir if probe_mode == "demo" else content_dir

    if not dir_exists_megfile(eval_dir):
        print(f"[SKIP] {model_dir}: 缺少 eval_images")
        return
    if not dir_exists_megfile(probe_dir):
        print(f"[SKIP] {model_dir}: 缺少 probe_dir={probe_dir} (probe_mode={probe_mode})")
        return
    if (not overwrite) and smart_exists(out_json):
        print(f"[SKIP] {model_dir}: {out_json} 已存在（未指定 --overwrite）")
        return

    ref_imgs_all = sorted(list_images_recursive_megfile(probe_dir))
    eval_imgs_all = sorted(list_images_recursive_megfile(eval_dir))

    if not ref_imgs_all:
        print(f"[WARN] {model_dir}: refs 为空：{probe_dir}")
        return
    if not eval_imgs_all:
        print(f"[WARN] {model_dir}: eval 为空：{eval_dir}")
        return

    # refs 用于提 common：采样
    ref_imgs = select_evenly_spaced(ref_imgs_all, max_refs) if (max_refs > 0) else ref_imgs_all
    # eval 可选采样
    eval_imgs = select_evenly_spaced(eval_imgs_all, max_eval) if (max_eval and max_eval > 0 and len(eval_imgs_all) > max_eval) else eval_imgs_all

    print(f"[INFO] {model_dir}: refs_used={len(ref_imgs)}/{len(ref_imgs_all)}, eval_used={len(eval_imgs)}/{len(eval_imgs_all)}")

    # 1) COMMON
    common_raw = ""
    common_text = None
    try:
        resp_common = client.chat.completions.create(
            model=model_name,
            messages=build_messages_for_common(ref_imgs, img_max_side, jpeg_quality),
            max_tokens=common_max_tokens,
            temperature=0.0,
        )
        common_raw = (resp_common.choices[0].message.content or "").strip()
        common_text = parse_common(common_raw)
    except Exception as e:
        print(f"[ERROR] {model_dir}: 提取 COMMON 失败 -> {e}")

    if not common_text:
        common_text = "无明显共同主题"
        print(f"[WARN] {model_dir}: COMMON 解析失败，回退为：{common_text}")
    else:
        print(f"[OK] {model_dir}: COMMON = {common_text}")

    # 2) eval 打分
    per_eval_similarity: Dict[str, float] = {}
    per_eval_label: Dict[str, int] = {}
    per_eval_conf: Dict[str, float] = {}
    per_eval_raw: Dict[str, str] = {}

    total, valid = 0.0, 0

    # clamp top_logprobs to [1, 20]
    top_lp = int(top_logprobs)
    if top_lp > 20:
        top_lp = 20
    if top_lp < 1:
        top_lp = 1

    for g_path in eval_imgs:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=build_messages_for_label(common_text, g_path, img_max_side, jpeg_quality),
                max_tokens=label_max_tokens,
                temperature=0.0,
                logprobs=True,
                top_logprobs=top_lp,
            )
            raw = (resp.choices[0].message.content or "").strip()
            label = parse_label(raw)
            conf = extract_label_confidence(resp, label)

            if label is None:
                print(f"[WARN] {model_dir}: eval={g_path} LABEL 解析失败，raw={raw!r}")
                continue
            if conf is None:
                # 兜底，不至于断分；你也可以改成 continue
                conf = 1.0

            sim = float(label) * float(conf)

            per_eval_similarity[g_path] = sim
            per_eval_label[g_path] = int(label)
            per_eval_conf[g_path] = float(conf)
            if save_raw:
                per_eval_raw[g_path] = raw

            total += sim
            valid += 1
            print(f"[OK] {model_dir}: eval={g_path}, label={label}, conf={conf:.4f}, sim={sim:.4f}")

        except Exception as e:
            print(f"[ERROR] {model_dir}: eval={g_path} 判别失败 -> {e}")
            continue

    if valid == 0:
        print(f"[WARN] {model_dir}: 无成功 eval 打分，跳过写入")
        return

    overall_mean = total / valid

    final_result = {
        "backend": "qwen_common_binary",
        "model": model_name,
        "probe_mode": probe_mode,
        "probe_dir": probe_dir,
        "demo_dir": demo_dir,
        "content_dir": content_dir,
        "eval_dir": eval_dir,
        "content_dir_name": content_dir_name,

        "system_prompt": SYSTEM_PROMPT,
        "common_prompt": COMMON_ONLY_PROMPT,
        "label_prompt_template": LABEL_ONLY_PROMPT_TEMPLATE,

        "common_content": common_text,
        "common_raw": common_raw if save_raw else None,

        "num_ref_images_total": len(ref_imgs_all),
        "num_ref_images_used_for_common": len(ref_imgs),
        "num_eval_images_total": len(eval_imgs_all),
        "num_eval_images_scored": valid,

        "per_eval_similarity": per_eval_similarity,
        "per_eval_label": per_eval_label,
        "per_eval_confidence": per_eval_conf,

        "overall_mean_similarity": overall_mean,
    }

    if save_raw:
        final_result["per_eval_raw"] = per_eval_raw

    # 去掉 None
    final_result = {k: v for k, v in final_result.items() if v is not None}

    out_dir = os.path.dirname(out_json)
    if out_dir:
        smart_makedirs(out_dir, exist_ok=True)
    with mopen(out_json, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"[OK] {model_dir}: overall_mean_similarity={overall_mean:.4f} -> {out_json}")


# ==================== worker ====================

def worker_main(worker_id: int, model_dirs: List[str], args_dict: dict) -> None:
    if not model_dirs:
        print(f"[WORKER-{worker_id}] 无任务，退出")
        return

    client = OpenAI(
        api_key=args_dict["api_key"],
        base_url=args_dict["base_url"],
        timeout=args_dict["timeout"],
    )

    print(f"[WORKER-{worker_id}] start: dirs={len(model_dirs)} model={args_dict['model']} probe_mode={args_dict['probe_mode']}")

    for model_dir in model_dirs:
        try:
            process_single_model_dir(
                model_dir=model_dir,
                client=client,
                model_name=args_dict["model"],
                output_name=args_dict["output_name"],
                overwrite=args_dict["overwrite"],
                probe_mode=args_dict["probe_mode"],
                content_dir_name=args_dict["content_dir_name"],
                max_refs=args_dict["max_refs"],
                max_eval=args_dict["max_eval"],
                img_max_side=args_dict["img_max_side"],
                jpeg_quality=args_dict["jpeg_quality"],
                common_max_tokens=args_dict["common_max_tokens"],
                label_max_tokens=args_dict["label_max_tokens"],
                top_logprobs=args_dict["top_logprobs"],
                save_raw=args_dict["save_raw"],
            )
        except Exception as e:
            print(f"[WORKER-{worker_id}] [ERROR] {model_dir}: {e}")


# ==================== main ====================

def main():
    parser = argparse.ArgumentParser(description="简化多进程：Qwen-VL common+binary 打分（server 推理 + 严格 system/prompt）")

    parser.add_argument("--root", required=True, help="根目录（本地或 s3://）")
    parser.add_argument("--output-name", default="content_similarity.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8, help="进程数")

    parser.add_argument("--probe-mode", choices=["demo", "content"], default="demo")
    parser.add_argument("--content-dir-name", type=str, default="content_100")

    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--timeout", type=int, default=600)

    parser.add_argument("--max-refs", type=int, default=8, help="用于提 COMMON 的 refs 最大张数（均匀采样）")
    parser.add_argument("--max-eval", type=int, default=0, help="可选：限制 eval 打分张数（0 不限制）")

    parser.add_argument("--img-max-side", type=int, default=1024, help="图片最长边 resize（0 不 resize）")
    parser.add_argument("--jpeg-quality", type=int, default=90)

    parser.add_argument("--common-max-tokens", type=int, default=96)
    parser.add_argument("--label-max-tokens", type=int, default=4, help="LABEL 只输出 0/1，一般 1-4 足够")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--save-raw", action="store_true")

    # 兼容旧参数（不再使用）
    parser.add_argument("--backend", default="qwen", help="兼容旧参数：已弃用")

    args = parser.parse_args()

    root = args.root.rstrip("/")
    model_dirs = iter_model_dirs_megfile(root)
    if not model_dirs:
        raise SystemExit(f"在 root={root} 下没有找到任何子目录")

    print(f"[INFO] found {len(model_dirs)} model dirs under {root}")

    num_workers = max(1, min(args.num_workers, len(model_dirs)))
    chunks = split_into_chunks(model_dirs, num_workers)
    num_workers = len(chunks)

    args_dict = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "timeout": args.timeout,
        "model": args.model,

        "output_name": args.output_name,
        "overwrite": args.overwrite,
        "probe_mode": args.probe_mode,
        "content_dir_name": args.content_dir_name,

        "max_refs": args.max_refs,
        "max_eval": args.max_eval,
        "img_max_side": args.img_max_side,
        "jpeg_quality": args.jpeg_quality,
        "common_max_tokens": args.common_max_tokens,
        "label_max_tokens": args.label_max_tokens,
        "top_logprobs": args.top_logprobs,
        "save_raw": args.save_raw,
    }

    print(f"[INFO] launch workers = {num_workers}")

    procs: List[mp.Process] = []
    for wid in range(num_workers):
        p = mp.Process(target=worker_main, args=(wid, chunks[wid], args_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("[DONE] all done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

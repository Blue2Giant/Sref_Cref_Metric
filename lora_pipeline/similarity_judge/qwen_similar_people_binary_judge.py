#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/LoraPipeline/sref_pipeline/qwen_similar_people_binary_judge.py \
  --root /mnt/jfs/loras_triplets/illustrious_0215_triplets_latest_9grid_all \
  --judge_times 3 \
  --min_true 2 \
  --model qwen3vlw8a8@ \
  --base_url http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1 \
  --out_all /data/LoraPipeline/assets/similar_people_all.json \
  --out_pos /data/LoraPipeline/assets/similar_people_bad.json \
  --out_neg /data/LoraPipeline/assets/similar_people_good.json \
  --out_detail /data/LoraPipeline/assets/similar_people_detail.json \
  --num_procs 32

python /data/LoraPipeline/sref_pipeline/qwen_similar_people_binary_judge.py \
  --root /mnt/jfs/loras_triplets/illustrious_0215_triplets_latest_9grid_all \
  --judge_times 3 \
  --min_true 2 \
  --model v1p3@ \
  --base_url http://stepcast-router.shai-core:9200/v1 \
  --out_all /data/LoraPipeline/assets/similar_people_all.json \
  --out_pos /data/LoraPipeline/assets/similar_people_bad.json \
  --out_neg /data/LoraPipeline/assets/similar_people_good.json \
  --out_detail /data/LoraPipeline/assets/similar_people_detail.json \
  --num_procs 32
"""
import os
import re
import json
import time
import base64
import argparse
import multiprocessing as mp
import shutil
import tempfile
from io import BytesIO
from typing import Dict, Optional, List, Tuple, Any, Iterable

import requests
from PIL import Image
from tqdm import tqdm

from megfile.smart import (
    smart_listdir,
    smart_exists,
    smart_open as mopen,
)

try:
    from megfile.smart import smart_isdir
except Exception:
    smart_isdir = None

API_KEY = "EMPTY"
MODEL = "v1p3"
BASE_URL = "http://stepcast-router.shai-core:9200/v1"
TIMEOUT = 180

RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85
Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def build_system_prompt(min_similar_people: int) -> str:
    min_similar_people = max(2, int(min_similar_people))
    if min_similar_people <= 2:
        return (
            "你是一个严格的图像审核器。\n"
            "任务：只判断图中是否出现“多个长得非常相似的人物”（如疑似同一人物复制、双胞胎式重复、克隆感）。\n"
            "只输出一个字符：0 或 1。\n"
            "1 表示：出现多个长得非常相似的人物。\n"
            "0 表示：没有出现该问题（包括只有1个人、多人但长相明显不同、无人像）。\n"
            "不要输出任何其他文字。"
        )

    return (
        "你是一个严格的图像审核器。\n"
        f"任务：只判断图中是否出现“至少 {min_similar_people} 个长得非常相似的人物”"
        "（如疑似同一人物复制、双胞胎式重复、克隆感）。\n"
        "只输出一个字符：0 或 1。\n"
        f"1 表示：出现至少 {min_similar_people} 个长得非常相似的人物。\n"
        f"0 表示：没有出现该问题（包括只有1个人、只有2个相似人物、相似人物数量少于 {min_similar_people} 个、"
        "多人但长相明显不同、无人像）。\n"
        "不要输出任何其他文字。"
    )


def build_user_instruction(min_similar_people: int) -> str:
    min_similar_people = max(2, int(min_similar_people))
    if min_similar_people <= 2:
        return (
            "请判断这张图是否存在“多个长得非常相似的人物”。\n"
            "判为 1 的情况：画面中至少两个人物在脸部特征、发型、年龄感、性别呈现、整体外观上高度相似，"
            "让人感觉像同一个人被复制或近似克隆。\n"
            "判为 0 的情况：\n"
            "1) 只有一个人物；\n"
            "2) 虽有多人但明显是不同人物；\n"
            "3) 没有人物主体。\n"
            "只输出 0 或 1。"
        )

    return (
        f"请判断这张图是否存在“至少 {min_similar_people} 个长得非常相似的人物”。\n"
        f"判为 1 的情况：画面中至少有 {min_similar_people} 个人物在脸部特征、发型、年龄感、性别呈现、整体外观上高度相似，"
        "让人感觉像同一个人被复制或近似克隆。\n"
        "判为 0 的情况：\n"
        f"1) 相似人物数量少于 {min_similar_people} 个（包括只有1个人，或只有2个高度相似的人）；\n"
        "2) 虽有多人但明显是不同人物；\n"
        "3) 没有人物主体。\n"
        "只输出 0 或 1。"
    )


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_dir(p: str) -> str:
    return p if p.endswith("/") else (p + "/")


def join_path(base: str, name: str) -> str:
    return base + name if base.endswith("/") else base + "/" + name


def is_remote(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("oss://")


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        if is_remote(path):
            with mopen(path, "rb") as f:
                return f.read()
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


def _probe_single_url(url: str, timeout_sec: float) -> Tuple[bool, str]:
    test_urls = [
        url.rstrip("/") + "/models",
        url.rstrip("/") + "/health",
        url.rstrip("/"),
    ]
    for u in test_urls:
        try:
            resp = requests.get(u, timeout=timeout_sec)
            if 200 <= int(resp.status_code) < 500:
                return True, u
        except Exception:
            continue
    return False, ""


def probe_endpoints(candidates: List[Tuple[str, str]], timeout_sec: float) -> List[Tuple[str, str]]:
    ok_eps: List[Tuple[str, str]] = []
    for model, url in candidates:
        alive, hit = _probe_single_url(url, timeout_sec=timeout_sec)
        if alive:
            log(f"[Host][OK] model={model} url={url} probe={hit}")
            ok_eps.append((model, url))
        else:
            log(f"[Host][DOWN] model={model} url={url}")
    return ok_eps


def call_qwen_chat_raw(
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 1,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "model": MODEL,
        "temperature": float(temperature),
        "messages": messages,
        "max_tokens": int(max_tokens),
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


def direct_judge_similar_people_01(
    image_path: str,
    min_similar_people: int = 2,
) -> Tuple[Optional[bool], str]:
    data = get_image_data_uri(image_path)
    if not data:
        return None, "图片编码失败"

    system_prompt = build_system_prompt(min_similar_people)
    user_instruction = build_user_instruction(min_similar_people)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instruction},
                {"type": "image_url", "image_url": {"url": data}},
                {"type": "text", "text": "只输出 0 或 1。"},
            ],
        },
    ]

    resp_json = call_qwen_chat_raw(
        messages,
        temperature=0.0,
        max_tokens=1,
    )
    if not resp_json:
        return None, "API 无响应/请求失败"

    choices = resp_json.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return None, "返回结构异常(无 choices)"

    text = strip_code_fences(_extract_text_from_choice(choices[0])).strip()
    pred_char = None
    for ch in text:
        if not ch.isspace():
            pred_char = ch
            break
    if pred_char not in ("0", "1"):
        return None, f"输出不是 0/1 (got={text!r})"

    return pred_char == "1", f"pred={pred_char}"


def judge_with_votes(
    image_path: str,
    judge_times: int,
    min_true: int,
    min_similar_people: int,
) -> Tuple[bool, Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    true_cnt = 0
    valid_cnt = 0
    false_cnt = 0

    for i in range(1, judge_times + 1):
        pred, reason = direct_judge_similar_people_01(
            image_path=image_path,
            min_similar_people=min_similar_people,
        )
        ok = isinstance(pred, bool)
        if ok:
            valid_cnt += 1
        if ok and pred is True:
            true_cnt += 1
        elif ok and pred is False:
            false_cnt += 1
        trials.append(
            {
                "call": i,
                "pred": pred,
                "valid": ok,
                "reason": reason,
            }
        )

    bad = true_cnt >= min_true
    status = "error" if valid_cnt <= 0 else ("true" if bad else "false")
    detail = {
        "image": image_path,
        "judge_times": judge_times,
        "min_true": min_true,
        "min_similar_people": min_similar_people,
        "true_cnt": true_cnt,
        "false_cnt": false_cnt,
        "valid_cnt": valid_cnt,
        "invalid_cnt": max(0, int(judge_times) - valid_cnt),
        "bad": bad,
        "status": status,
        "trials": trials,
    }
    return bad, detail


def smart_write_json(path: str, obj: Any):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    if is_remote(path):
        with mopen(path, "wb") as f:
            f.write(data)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def smart_write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    if is_remote(path):
        with mopen(path, "wb") as f:
            f.write(text.encode("utf-8"))
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def _open_binary_writer(path: str):
    if is_remote(path):
        return mopen(path, "wb")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return open(path, "wb")


class JsonObjectStreamWriter:
    def __init__(self, path: str):
        self.path = path
        self.fp = _open_binary_writer(path)
        self.first = True
        self.closed = False
        self.fp.write(b"{\n")

    def write_item(self, key: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError(f"writer already closed: {self.path}")
        if not self.first:
            self.fp.write(b",\n")
        chunk = f"  {json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
        self.fp.write(chunk.encode("utf-8"))
        self.first = False

    def close(self) -> None:
        if self.closed:
            return
        if not self.first:
            self.fp.write(b"\n")
        self.fp.write(b"}\n")
        self.fp.close()
        self.closed = True


class JsonlStreamWriter:
    def __init__(self, path: str):
        self.path = path
        self.fp = _open_binary_writer(path)
        self.closed = False

    def write_row(self, row: Dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError(f"writer already closed: {self.path}")
        self.fp.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))

    def close(self) -> None:
        if self.closed:
            return
        self.fp.close()
        self.closed = True


class DetailStreamWriter:
    def __init__(self, path: str):
        self.path = path
        fd, temp_path = tempfile.mkstemp(prefix="qwen_similar_people_detail_", suffix=".json.part")
        os.close(fd)
        self.temp_path = temp_path
        self.temp_fp = open(self.temp_path, "w", encoding="utf-8")
        self.first = True
        self.closed = False

    def write_result(self, rec: Dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError(f"writer already closed: {self.path}")
        if not self.first:
            self.temp_fp.write(",\n")
        self.temp_fp.write(json.dumps(rec, ensure_ascii=False))
        self.first = False

    def finalize(self, summary: Dict[str, Any]) -> None:
        if self.closed:
            return
        self.temp_fp.close()
        out_fp = _open_binary_writer(self.path)
        try:
            out_fp.write(b"{\n  \"summary\": ")
            out_fp.write(json.dumps(summary, ensure_ascii=False).encode("utf-8"))
            out_fp.write(b",\n  \"results\": [")
            if not self.first:
                out_fp.write(b"\n")
                with open(self.temp_path, "rb") as src_fp:
                    shutil.copyfileobj(src_fp, out_fp)
                out_fp.write(b"\n  ")
            out_fp.write(b"]\n}\n")
        finally:
            out_fp.close()
            try:
                os.remove(self.temp_path)
            except OSError:
                pass
            self.closed = True


def _strip_prefix(path: str, root: str) -> str:
    r = root.rstrip("/")
    if path == r:
        return ""
    if path.startswith(r + "/"):
        return path[len(r) + 1 :]
    return path


def _iter_remote_images(root: str) -> Iterable[str]:
    stack = [root.rstrip("/")]
    while stack:
        cur = stack.pop()
        try:
            names = smart_listdir(cur)
        except Exception:
            if smart_exists(cur) and is_image_name(os.path.basename(cur)):
                yield cur
            continue
        for name in names:
            raw = str(name).rstrip("/")
            if raw.startswith("s3://") or raw.startswith("oss://"):
                full = raw
            elif raw.startswith(cur.rstrip("/") + "/"):
                full = raw
            else:
                full = join_path(cur, raw)

            is_dir = False
            if smart_isdir is not None:
                try:
                    is_dir = bool(smart_isdir(full))
                except Exception:
                    is_dir = False
            if not is_dir:
                try:
                    _ = smart_listdir(full)
                    is_dir = True
                except Exception:
                    is_dir = False
            if is_dir:
                stack.append(full.rstrip("/"))
            else:
                if is_image_name(os.path.basename(full)):
                    yield full


def iter_all_images(root: str) -> List[str]:
    out: List[str] = []
    if is_remote(root):
        for p in _iter_remote_images(root):
            out.append(p)
    else:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if is_image_name(fn):
                    out.append(os.path.join(dp, fn))
    out.sort()
    return out


def collect_tasks_from_jsonl(
    path: str,
    allowed_keys: Optional[set] = None,
    max_images_per_key: int = 0,
    task_mode: str = "per_value",
) -> List[Dict[str, Any]]:
    if task_mode == "per_value":
        tasks: List[Dict[str, Any]] = []
        seen = set()
        with mopen(path, "r", encoding="utf-8") as f:
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
                    key_name = str(k).strip()
                    if not key_name:
                        continue
                    if allowed_keys is not None and key_name not in allowed_keys:
                        continue
                    vals: List[str] = []
                    if isinstance(v, str):
                        vals = [v]
                    elif isinstance(v, list):
                        vals = [x for x in v if isinstance(x, str)]
                    for p in vals:
                        if not is_image_name(p):
                            continue
                        item_key = p
                        if item_key in seen:
                            continue
                        seen.add(item_key)
                        tasks.append({"key": item_key, "image_paths": [p]})
        if max_images_per_key > 0:
            tasks = tasks[:max_images_per_key]
        return tasks

    merged: Dict[str, List[str]] = {}
    with mopen(path, "r", encoding="utf-8") as f:
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
                key = str(k).strip()
                if not key:
                    continue
                if allowed_keys is not None and key not in allowed_keys:
                    continue
                arr = merged.setdefault(key, [])
                if isinstance(v, str):
                    if is_image_name(v):
                        arr.append(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, str) and is_image_name(x):
                            arr.append(x)
    tasks: List[Dict[str, Any]] = []
    for k, imgs in merged.items():
        uniq = []
        seen = set()
        for p in imgs:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
            if max_images_per_key > 0 and len(uniq) >= max_images_per_key:
                break
        if uniq:
            tasks.append({"key": k, "image_paths": uniq})
    tasks.sort(key=lambda x: x["key"])
    return tasks


def make_key(path: str, root: str) -> str:
    rel = _strip_prefix(path, root).replace("\\", "/")
    rel_no_ext = os.path.splitext(rel)[0]
    k = re.sub(r"[^\w/.-]+", "_", rel_no_ext)
    return k


POOL_ROOT = ""
POOL_JUDGE_TIMES = 3
POOL_MIN_TRUE = 2
POOL_MIN_SIMILAR_PEOPLE = 2
POOL_JSONL_MODE = False


def _process_one_for_pool(path: str) -> Dict[str, Any]:
    bad, detail = judge_with_votes(
        image_path=path,
        judge_times=max(1, int(POOL_JUDGE_TIMES)),
        min_true=max(1, int(POOL_MIN_TRUE)),
        min_similar_people=max(2, int(POOL_MIN_SIMILAR_PEOPLE)),
    )
    status = str(detail.get("status", "true" if bad else "false"))
    return {
        "path": path,
        "key": make_key(path, POOL_ROOT),
        "bad_similar_people": bool(bad),
        "status": status,
        "detail": detail,
    }


def _process_key_task_for_pool(task: Dict[str, Any]) -> Dict[str, Any]:
    key = str(task.get("key", ""))
    paths = task.get("image_paths", []) or []
    per_image: List[Dict[str, Any]] = []
    bad_any = False
    true_paths: List[str] = []
    false_paths: List[str] = []
    error_paths: List[str] = []
    for p in paths:
        bad, detail = judge_with_votes(
            image_path=p,
            judge_times=max(1, int(POOL_JUDGE_TIMES)),
            min_true=max(1, int(POOL_MIN_TRUE)),
            min_similar_people=max(2, int(POOL_MIN_SIMILAR_PEOPLE)),
        )
        status = str(detail.get("status", "true" if bad else "false"))
        bad = bool(bad)
        if status == "true":
            bad_any = True
            true_paths.append(p)
        elif status == "false":
            false_paths.append(p)
        else:
            error_paths.append(p)
        per_image.append({"path": p, "status": status, "bad": bad, "detail": detail})
    return {
        "path": "",
        "key": key,
        "bad_similar_people": bad_any,
        "status": "true" if bad_any else ("error" if error_paths and not false_paths else "false"),
        "detail": {
            "mode": "jsonl_key_aggregate",
            "key": key,
            "image_count": len(paths),
            "bad_any": bad_any,
            "true_paths": true_paths,
            "false_paths": false_paths,
            "error_paths": error_paths,
            "per_image": per_image,
        },
    }


def smart_write_txt_lines(path: str, lines: List[str]) -> None:
    text = "\n".join(lines)
    if text:
        text += "\n"
    if path.startswith("s3://") or path.startswith("oss://"):
        with mopen(path, "wb") as f:
            f.write(text.encode("utf-8"))
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def dedupe_keep_order(paths: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for path in paths:
        if not isinstance(path, str):
            continue
        p = path.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def extract_path_buckets_for_record(rec: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[str]]:
    key = str(rec.get("key", "")).strip()
    detail = rec.get("detail", {}) if isinstance(rec.get("detail"), dict) else {}
    if detail.get("mode") == "jsonl_key_aggregate":
        true_paths = dedupe_keep_order(list(detail.get("true_paths", []) or []))
        false_paths = dedupe_keep_order(list(detail.get("false_paths", []) or []))
        error_paths = dedupe_keep_order(list(detail.get("error_paths", []) or []))
        return key, true_paths, false_paths, error_paths

    path = str(rec.get("path", "")).strip()
    status = str(rec.get("status") or detail.get("status") or ("true" if rec.get("bad_similar_people") else "false"))
    true_paths = [path] if path and status == "true" else []
    false_paths = [path] if path and status == "false" else []
    error_paths = [path] if path and status == "error" else []
    return key, true_paths, false_paths, error_paths


def build_path_bucket_rows(
    results: List[Dict[str, Any]],
    keep_empty_keys: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    true_rows: List[Dict[str, Any]] = []
    false_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for rec in results:
        key, true_paths, false_paths, error_paths = extract_path_buckets_for_record(rec)
        if not key:
            continue

        if keep_empty_keys or true_paths:
            true_rows.append({key: true_paths})
        if keep_empty_keys or false_paths:
            false_rows.append({key: false_paths})
        if keep_empty_keys or error_paths:
            error_rows.append({key: error_paths})

    return true_rows, false_rows, error_rows


def iter_processed_results(
    picked_tasks: List[Dict[str, Any]],
    picked: List[str],
    num_procs: int,
) -> Iterable[Dict[str, Any]]:
    total_tasks = len(picked_tasks) if picked_tasks else len(picked)
    progress_desc = "SimilarPeopleKey" if picked_tasks else "SimilarPeopleImage"
    progress_unit = "key" if picked_tasks else "image"

    if num_procs and num_procs > 1:
        procs = max(1, int(num_procs))
        log(f"[Info] 使用多进程 num_procs={procs}")
        with mp.Pool(processes=procs) as pool:
            if picked_tasks:
                iterator = pool.imap(_process_key_task_for_pool, picked_tasks, chunksize=1)
            else:
                iterator = pool.imap(_process_one_for_pool, picked, chunksize=1)
            with tqdm(total=total_tasks, desc=progress_desc, unit=progress_unit) as pbar:
                for rec in iterator:
                    pbar.update(1)
                    yield rec
        return

    with tqdm(total=total_tasks, desc=progress_desc, unit=progress_unit) as pbar:
        if picked_tasks:
            for i, task in enumerate(picked_tasks, 1):
                rec = _process_key_task_for_pool(task)
                pbar.update(1)
                log(f"[{i}/{len(picked_tasks)}] {task['key']} -> bad={rec['bad_similar_people']}")
                yield rec
        else:
            for i, p in enumerate(picked, 1):
                rec = _process_one_for_pool(p)
                pbar.update(1)
                log(f"[{i}/{len(picked)}] {os.path.basename(p)} -> bad={rec['bad_similar_people']}")
                yield rec


def main():
    global MODEL, BASE_URL, POOL_ROOT, POOL_JUDGE_TIMES, POOL_MIN_TRUE, POOL_MIN_SIMILAR_PEOPLE, POOL_JSONL_MODE

    ap = argparse.ArgumentParser("判别图片是否出现多个长得相似的人物（0/1）")
    ap.add_argument("--root", default="", help="待判别图片根目录（本地或 s3://）")
    ap.add_argument("--input-jsonl", default="", help="可选：输入 jsonl，每行对象，value 为图片路径或图片路径列表")
    ap.add_argument("--jsonl-keys", default="", help="可选：jsonl 模式仅处理这些 key（逗号分隔）")
    ap.add_argument("--jsonl-task-mode", choices=["per_value", "aggregate_key"], default="per_value", help="jsonl 任务模式：按每个图片值判别，或按 key 聚合判别")
    ap.add_argument("--max_images_per_key", type=int, default=0, help="jsonl 模式每个 key 最多处理多少张图，<=0 表示不限制")
    ap.add_argument("--num_samples", type=int, default=0, help="抽样数量；<=0 表示全量")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--judge_times", type=int, default=3, help="每张图判别次数")
    ap.add_argument("--min_true", type=int, default=2, help="至少多少次判为1，最终才判为1")
    ap.add_argument("--min_similar_people", type=int, default=2, help="至少多少个高度相似的人物同时出现，最终才判为1")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--base_url", type=str, default=BASE_URL)
    ap.add_argument("--endpoint", action="append", default=[], help="可重复传入 model@url 或 url，自动探测可用 endpoint")
    ap.add_argument("--probe-timeout", type=float, default=3.0, help="endpoint 探测超时秒数")
    ap.add_argument("--out_all", required=True, help="全量 map: key->0/1，1表示有多个相似人物")
    ap.add_argument("--out_pos", required=True, help="正样本 map: 仅 value=1（坏图）")
    ap.add_argument("--out_neg", required=True, help="负样本 map: 仅 value=0（可保留）")
    ap.add_argument("--out_true_jsonl", default="", help="可选：jsonl 模式下判为真的路径列表输出")
    ap.add_argument("--out_false_jsonl", default="", help="可选：jsonl 模式下判为假的路径列表输出")
    ap.add_argument("--out_error_jsonl", default="", help="可选：jsonl 模式下判别失败路径列表输出")
    ap.add_argument("--keep_empty_keys", action="store_true", help="jsonl 路径列表输出时保留空列表 key")
    ap.add_argument("--out_pos_txt", default="", help="可选：坏图 key 列表 txt（一行一个）")
    ap.add_argument("--out_neg_txt", default="", help="可选：好图 key 列表 txt（一行一个）")
    ap.add_argument("--out_detail", default="", help="可选详细结果 JSON")
    ap.add_argument("--num_procs", type=int, default=0, help="多进程 worker 数；0 或 1 表示单进程")
    args = ap.parse_args()

    if (args.out_true_jsonl or args.out_false_jsonl or args.out_error_jsonl) and not args.input_jsonl:
        raise RuntimeError("路径列表 jsonl 输出仅支持 --input-jsonl 模式")

    endpoints: List[Tuple[str, str]] = []
    for e in args.endpoint:
        s = str(e).strip()
        if not s:
            continue
        if "@" in s:
            model, url = s.split("@", 1)
            endpoints.append((model.strip(), url.strip()))
        else:
            endpoints.append((args.model, s))
    if not endpoints:
        endpoints = [(args.model, args.base_url)]
    log("[Host] candidates:")
    for model, url in endpoints:
        log(f"[Host] candidate model={model} url={url}")
    endpoints = probe_endpoints(endpoints, timeout_sec=max(0.5, float(args.probe_timeout)))
    if not endpoints:
        raise RuntimeError("没有可用endpoint，探测全部失败")
    MODEL, BASE_URL = endpoints[0]
    log(f"[Host] selected model={MODEL} url={BASE_URL}")

    key_tasks: List[Dict[str, Any]] = []
    candidates: List[str] = []
    if args.input_jsonl:
        allow = None
        if args.jsonl_keys.strip():
            allow = {x.strip() for x in args.jsonl_keys.split(",") if x.strip()}
        key_tasks = collect_tasks_from_jsonl(
            args.input_jsonl,
            allowed_keys=allow,
            max_images_per_key=max(0, int(args.max_images_per_key)),
            task_mode=args.jsonl_task_mode,
        )
        if not key_tasks:
            raise RuntimeError(f"{args.input_jsonl} 中未解析到有效图片路径")
        total_images = sum(len(t.get("image_paths", [])) for t in key_tasks)
        log(f"[Info] jsonl keys={len(key_tasks)} total_images={total_images} max_images_per_key={args.max_images_per_key}")
        preview_limit = 20
        for t in key_tasks[:preview_limit]:
            log(f"[Info] key={t['key']} images={len(t.get('image_paths', []))}")
        if len(key_tasks) > preview_limit:
            log(f"[Info] ... omitted {len(key_tasks) - preview_limit} more keys")
    else:
        if not args.root:
            raise RuntimeError("未提供 --input-jsonl 时，必须提供 --root")
        candidates = iter_all_images(args.root)
        if not candidates:
            raise RuntimeError(f"{args.root} 下没找到图片")

    import random

    rng = random.Random(args.seed)
    if key_tasks:
        if args.num_samples and args.num_samples > 0 and args.num_samples < len(key_tasks):
            picked_tasks = rng.sample(key_tasks, args.num_samples)
        else:
            picked_tasks = key_tasks
        picked = []
        log(f"[Info] picked_keys={len(picked_tasks)} from keys={len(key_tasks)}")
    else:
        if args.num_samples and args.num_samples > 0 and args.num_samples < len(candidates):
            picked = rng.sample(candidates, args.num_samples)
        else:
            picked = candidates
        picked_tasks = []
        log(f"[Info] picked={len(picked)} from candidates={len(candidates)}")

    POOL_ROOT = args.root
    POOL_JUDGE_TIMES = max(1, int(args.judge_times))
    POOL_MIN_TRUE = max(1, int(args.min_true))
    POOL_MIN_SIMILAR_PEOPLE = max(2, int(args.min_similar_people))
    POOL_JSONL_MODE = bool(args.input_jsonl)
    write_path_bucket_jsonls = bool(
        args.input_jsonl and (args.out_true_jsonl or args.out_false_jsonl or args.out_error_jsonl)
    )

    all_writer = JsonObjectStreamWriter(args.out_all)
    pos_writer = JsonObjectStreamWriter(args.out_pos)
    neg_writer = JsonObjectStreamWriter(args.out_neg)
    true_writer = JsonlStreamWriter(args.out_true_jsonl) if args.out_true_jsonl else None
    false_writer = JsonlStreamWriter(args.out_false_jsonl) if args.out_false_jsonl else None
    error_writer = JsonlStreamWriter(args.out_error_jsonl) if args.out_error_jsonl else None
    detail_writer = DetailStreamWriter(args.out_detail) if args.out_detail else None

    pos_keys: List[str] = []
    neg_keys: List[str] = []
    total_count = 0
    bad_count = 0
    good_count = 0
    error_key_count = 0
    error_image_count = 0

    try:
        for rec in iter_processed_results(
            picked_tasks=picked_tasks,
            picked=picked,
            num_procs=int(args.num_procs),
        ):
            k = str(rec.get("key", "")).strip()
            if not k:
                continue
            key, true_paths, false_paths, error_paths = extract_path_buckets_for_record(rec)
            v = 1 if true_paths else 0

            total_count += 1
            if v == 1:
                bad_count += 1
                pos_keys.append(k)
            else:
                good_count += 1
                neg_keys.append(k)

            if error_paths:
                error_key_count += 1
                error_image_count += len(error_paths)

            all_writer.write_item(k, v)
            if v == 1:
                pos_writer.write_item(k, 1)
            else:
                neg_writer.write_item(k, 0)

            if write_path_bucket_jsonls:
                if true_writer and (args.keep_empty_keys or true_paths):
                    true_writer.write_row({key: true_paths})
                if false_writer and (args.keep_empty_keys or false_paths):
                    false_writer.write_row({key: false_paths})
                if error_writer and (args.keep_empty_keys or error_paths):
                    error_writer.write_row({key: error_paths})

            if detail_writer:
                detail_writer.write_result(rec)
    finally:
        all_writer.close()
        pos_writer.close()
        neg_writer.close()
        if true_writer:
            true_writer.close()
        if false_writer:
            false_writer.close()
        if error_writer:
            error_writer.close()

    if args.out_pos_txt:
        smart_write_txt_lines(args.out_pos_txt, sorted(pos_keys))
        log(f"  -> {args.out_pos_txt}")
    if args.out_neg_txt:
        smart_write_txt_lines(args.out_neg_txt, sorted(neg_keys))
        log(f"  -> {args.out_neg_txt}")

    log(f"[DONE] all={total_count} bad={bad_count} good={good_count}")
    log(f"  -> {args.out_all}")
    log(f"  -> {args.out_pos}")
    log(f"  -> {args.out_neg}")

    if true_writer:
        log(f"  -> {args.out_true_jsonl}")
    if false_writer:
        log(f"  -> {args.out_false_jsonl}")
    if error_writer:
        log(f"  -> {args.out_error_jsonl}")

    if detail_writer:
        summary = {
            "root": args.root,
            "input_jsonl": args.input_jsonl,
            "picked": len(picked_tasks) if picked_tasks else len(picked),
            "bad_count": bad_count,
            "bad_ratio": bad_count / float(len(picked_tasks) if picked_tasks else len(picked)) if (picked_tasks or picked) else 0.0,
            "error_key_count": error_key_count,
            "error_image_count": error_image_count,
            "judge_times": args.judge_times,
            "min_true": args.min_true,
            "min_similar_people": args.min_similar_people,
            "model": MODEL,
            "base_url": BASE_URL,
            "out_true_jsonl": args.out_true_jsonl,
            "out_false_jsonl": args.out_false_jsonl,
            "out_error_jsonl": args.out_error_jsonl,
        }
        detail_writer.finalize(summary)
        log(f"[Detail] -> {args.out_detail}")


if __name__ == "__main__":
    main()

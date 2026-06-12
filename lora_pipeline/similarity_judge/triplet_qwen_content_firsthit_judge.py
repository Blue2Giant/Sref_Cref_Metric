#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import time
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm

import triplet_qwen_dual_judge as base
import qwen_judge_runtime as runtime


G_ARGS = None
PAIR_KEY_ORDER_CHOICES = ("content_style", "style_content")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_pair_key_ids(pair_key: str, pair_key_order: str) -> Tuple[str, str]:
    left_id, right_id = pair_key.split("__", 1)
    left_id = left_id.strip()
    right_id = right_id.strip()
    if pair_key_order == "content_style":
        return left_id, right_id
    if pair_key_order == "style_content":
        return right_id, left_id
    raise ValueError(f"未知的 pair_key_order: {pair_key_order}")


def _probe_single_url(url: str, timeout_sec: float) -> Tuple[bool, str]:
    test_urls = [
        url.rstrip("/") + "/models",
        url.rstrip("/") + "/health",
        url.rstrip("/"),
    ]
    for u in test_urls:
        try:
            resp = base.requests.get(u, timeout=timeout_sec)
            if 200 <= int(resp.status_code) < 500:
                return True, u
        except Exception:
            continue
    return False, ""


def probe_endpoints(candidates: List[Tuple[str, str]], timeout_sec: float) -> List[Tuple[str, str]]:
    ok_eps: List[Tuple[str, str]] = []
    for m, u in candidates:
        alive, hit = _probe_single_url(u, timeout_sec=timeout_sec)
        if alive:
            log(f"[Host][OK] model={m} url={u} probe={hit}")
            ok_eps.append((m, u))
        else:
            log(f"[Host][DOWN] model={m} url={u}")
    return ok_eps


def _is_remote_path(path: str) -> bool:
    s = str(path or "")
    return s.startswith("s3://") or s.startswith("oss://")


def _smart_join(base_dir: str, name: str) -> str:
    if _is_remote_path(base_dir):
        return base_dir.rstrip("/") + "/" + str(name).lstrip("/")
    return os.path.join(base_dir, name)


def _sanitize_name(s: str) -> str:
    x = str(s or "").strip()
    return x.replace("/", "_").replace("\\", "_")


def _smart_copy_file(src: str, dst: str):
    dst_dir = os.path.dirname(dst) if not _is_remote_path(dst) else dst.rsplit("/", 1)[0]
    if dst_dir:
        base.smart_makedirs(dst_dir, exist_ok=True)
    with base.mopen(src, "rb") as fin:
        data = fin.read()
    with base.mopen(dst, "wb") as fout:
        fout.write(data)


def copy_from_result_jsonl(copy_from_jsonl: str, copy_out_dir: str, copy_max_items: int = 0):
    if not copy_from_jsonl:
        raise RuntimeError("copy_from_jsonl 不能为空")
    if not copy_out_dir:
        raise RuntimeError("copy_out_dir 不能为空")
    if not base.smart_exists(copy_from_jsonl):
        raise RuntimeError(f"copy_from_jsonl 不存在: {copy_from_jsonl}")
    base.smart_makedirs(copy_out_dir, exist_ok=True)

    total_records = 0
    non_empty_records = 0
    copied_files = 0
    fail_files = 0
    with base.mopen(copy_from_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict) or len(obj) != 1:
                continue
            pair_key, value = next(iter(obj.items()))
            total_records += 1
            if not isinstance(value, list) or len(value) == 0:
                continue
            non_empty_records += 1
            key_dir = _smart_join(copy_out_dir, _sanitize_name(pair_key))
            for i, src in enumerate(value, 1):
                if not isinstance(src, str) or not src.strip():
                    continue
                src = src.strip()
                if not base.smart_exists(src):
                    fail_files += 1
                    log(f"[Copy][MISS] key={pair_key} src_not_found={src}")
                    continue
                if len(value) > 1:
                    dst_dir = _smart_join(key_dir, f"{i:03d}")
                else:
                    dst_dir = key_dir
                dst = _smart_join(dst_dir, os.path.basename(src))
                try:
                    _smart_copy_file(src, dst)
                    if base.smart_exists(dst):
                        copied_files += 1
                    else:
                        fail_files += 1
                        log(f"[Copy][FAIL] key={pair_key} dst_missing={dst}")
                except Exception as e:
                    fail_files += 1
                    log(f"[Copy][ERR] key={pair_key} src={src} dst={dst} err={e}")
            if copy_max_items > 0 and non_empty_records >= int(copy_max_items):
                break
    log(
        f"[Copy][DONE] total_records={total_records} non_empty_records={non_empty_records} "
        f"copied_files={copied_files} fail_files={fail_files} out_dir={copy_out_dir}"
    )


def read_content_index(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not os.path.isfile(path):
        raise RuntimeError(f"content索引文件不存在: {path}")
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
                if not isinstance(k, str) or not isinstance(v, list):
                    continue
                paths = [str(x).strip() for x in v if isinstance(x, str) and str(x).strip()]
                if paths:
                    out[k] = paths
    return out


def _dedupe_preserve_order(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for path in paths:
        s = str(path).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_triplet_jsonl(
    path: str,
    content_index: Dict[str, List[str]],
    pair_key_order: str = "content_style",
    per_image: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    tasks: List[Dict[str, Any]] = []
    skipped = 0
    if not os.path.isfile(path):
        raise RuntimeError(f"triplet文件不存在: {path}")
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
                if not isinstance(pair_key, str) or "__" not in pair_key:
                    skipped += 1
                    continue
                if not isinstance(arr, list) or not arr:
                    skipped += 1
                    continue
                main_img = str(arr[0]).strip()
                if per_image:
                    main_imgs = _dedupe_preserve_order([x for x in arr if isinstance(x, str)])
                else:
                    main_imgs = [main_img] if main_img else []
                if not main_imgs:
                    skipped += 1
                    continue
                content_id, _sid = parse_pair_key_ids(pair_key, pair_key_order=pair_key_order)
                content_id = content_id.strip()
                content_imgs = content_index.get(content_id, [])
                for main_img in main_imgs:
                    tasks.append(
                        {
                            "pair_key": pair_key,
                            "result_key": main_img if per_image else pair_key,
                            "main_img": main_img,
                            "content_id": content_id,
                            "content_imgs": content_imgs,
                        }
                    )
    return tasks, skipped


def load_existing_done(paths: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
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
                    if isinstance(v, list):
                        out[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, (str, dict)):
                        out[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
    return out


def decide_matched_paths_output(matched_paths: List[str], match_threshold: int) -> List[str]:
    if int(match_threshold) < 0:
        raise ValueError("match_threshold 必须 >= 0")
    if len(matched_paths) >= int(match_threshold):
        return list(matched_paths[: int(match_threshold)])
    return []


def sample_paths_for_all_similar(paths: Sequence[str], sample_size: int, seed: int, result_key: str) -> List[str]:
    uniq_paths = _dedupe_preserve_order(paths)
    if not uniq_paths:
        return []
    sample_size = max(1, int(sample_size))
    if len(uniq_paths) <= sample_size:
        return uniq_paths
    rng = random.Random(f"{seed}:{result_key}")
    return rng.sample(uniq_paths, sample_size)


def _new_endpoint_agg() -> Dict[str, float]:
    out: Dict[str, float] = {
        "tasks": 0,
        "task_wall_sec": 0.0,
        "matched": 0,
        "all_similar": 0,
        "no_match": 0,
        "errors": 0,
    }
    for key in runtime.RUNTIME_STAT_KEYS:
        out[key] = 0.0
    return out


def _clone_endpoint_aggs(endpoint_aggs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {k: dict(v) for k, v in endpoint_aggs.items()}


def _update_endpoint_agg(endpoint_aggs: Dict[str, Dict[str, float]], rec: Dict[str, Any]):
    endpoint = str(rec.get("endpoint", "") or "unknown")
    agg = endpoint_aggs.setdefault(endpoint, _new_endpoint_agg())
    agg["tasks"] += 1
    agg["task_wall_sec"] += float(rec.get("task_wall_sec", 0.0) or 0.0)
    bucket = str(rec.get("bucket", "") or "")
    if bucket == "matched":
        agg["matched"] += 1
    elif bucket == "all_similar":
        agg["all_similar"] += 1
    elif bucket == "no_match":
        agg["no_match"] += 1
    if rec.get("error"):
        agg["errors"] += 1
    runtime_stats = rec.get("runtime", {})
    if isinstance(runtime_stats, dict):
        for key in runtime.RUNTIME_STAT_KEYS:
            agg[key] += float(runtime_stats.get(key, 0.0) or 0.0)


def _log_endpoint_stats(
    endpoint_aggs: Dict[str, Dict[str, float]],
    total_elapsed_sec: float,
    window_elapsed_sec: float,
    prev_snapshot: Dict[str, Dict[str, float]],
):
    if not endpoint_aggs:
        return
    log(
        f"[EndpointStats] total_elapsed={max(0.0, total_elapsed_sec):.1f}s "
        f"window={max(0.0, window_elapsed_sec):.1f}s"
    )
    for endpoint in sorted(endpoint_aggs):
        cur = endpoint_aggs[endpoint]
        prev = prev_snapshot.get(endpoint, {})
        window_tasks = cur["tasks"] - float(prev.get("tasks", 0.0) or 0.0)
        window_calls = cur["api_calls"] - float(prev.get("api_calls", 0.0) or 0.0)
        avg_api_ms = (cur["api_elapsed_sec"] * 1000.0 / cur["api_calls"]) if cur["api_calls"] > 0 else 0.0
        cache_total = cur["cache_hits"] + cur["cache_misses"]
        cache_hit_rate = (cur["cache_hits"] / cache_total) if cache_total > 0 else 0.0
        window_qps = (window_calls / window_elapsed_sec) if window_elapsed_sec > 0 else 0.0
        log(
            f"[EndpointStats] endpoint={endpoint} "
            f"tasks={int(cur['tasks'])}(+{int(window_tasks)}) "
            f"api={int(cur['api_calls'])}(+{int(window_calls)}, {window_qps:.2f}/s) "
            f"ok={int(cur['api_success'])} fail={int(cur['api_fail'])} "
            f"retry_exhausted={int(cur['api_retry_exhausted'])} "
            f"avg_api_ms={avg_api_ms:.0f} cache_hit={cache_hit_rate:.1%} "
            f"matched={int(cur['matched'])} all_similar={int(cur['all_similar'])} "
            f"no_match={int(cur['no_match'])} err={int(cur['errors'])}"
        )


def _judge_one(task: Dict[str, Any]) -> Dict[str, Any]:
    args = G_ARGS
    pair_key = task["pair_key"]
    result_key = task.get("result_key", pair_key)
    main_img = task["main_img"]
    content_imgs = task["content_imgs"]
    match_threshold = int(args.match_threshold)
    exact_all_similar = bool(getattr(args, "exact_all_similar", False)) and bool(getattr(args, "per_image", False))
    refs_examined = 0
    early_stop = ""

    if not base.smart_exists(main_img):
        return {
            "pair_key": pair_key,
            "result_key": result_key,
            "value": [],
            "bucket": "error",
            "error": f"main_not_found: {main_img}",
            "refs_total": 0,
            "refs_examined": refs_examined,
            "early_stop": early_stop,
        }
    if not content_imgs:
        return {
            "pair_key": pair_key,
            "result_key": result_key,
            "value": [],
            "bucket": "error",
            "error": f"content_id_not_found: {task['content_id']}",
            "refs_total": 0,
            "refs_examined": refs_examined,
            "early_stop": early_stop,
        }

    existing_content_imgs = _dedupe_preserve_order([cp for cp in content_imgs if base.smart_exists(cp)])
    if not existing_content_imgs:
        return {
            "pair_key": pair_key,
            "result_key": result_key,
            "value": [],
            "bucket": "error",
            "error": f"content_refs_not_found: {task['content_id']}",
            "refs_total": 0,
            "refs_examined": refs_examined,
            "early_stop": early_stop,
        }

    matched_paths: List[str] = []
    refs_total = len(existing_content_imgs)
    for idx, cp in enumerate(existing_content_imgs, 1):
        decision, _detail, retry_exhausted = base.judge_pair_voting(
            path_a=main_img,
            path_b=cp,
            system_prompt=base.CONTENT_SYSTEM_PROMPT,
            user_instruction=base.CONTENT_USER_INSTRUCTION,
            conf_thr=float(args.content_conf_thr),
            judge_times=int(args.content_judge_times),
            min_true=int(args.content_min_true),
        )
        refs_examined = idx
        if retry_exhausted:
            return {
                "pair_key": pair_key,
                "result_key": result_key,
                "value": [],
                "bucket": "error",
                "error": "retry_exhausted",
                "refs_total": refs_total,
                "refs_examined": refs_examined,
                "early_stop": early_stop,
            }
        if decision is True:
            matched_paths.append(cp)
            if match_threshold > 0 and len(matched_paths) >= match_threshold and not exact_all_similar:
                early_stop = "match_threshold_hit"
                break
        remaining_refs = refs_total - refs_examined
        if match_threshold > 0 and len(matched_paths) + remaining_refs < match_threshold:
            early_stop = "match_threshold_unreachable"
            break

    if bool(getattr(args, "per_image", False)):
        if (not matched_paths) or (len(matched_paths) < match_threshold):
            return {
                "pair_key": pair_key,
                "result_key": result_key,
                "value": [],
                "bucket": "no_match",
                "error": "",
                "refs_total": refs_total,
                "refs_examined": refs_examined,
                "early_stop": early_stop,
            }
        if exact_all_similar and refs_examined >= refs_total and len(matched_paths) == refs_total:
            sampled_paths = sample_paths_for_all_similar(
                matched_paths,
                sample_size=int(args.all_similar_sample_size),
                seed=int(args.seed),
                result_key=result_key,
            )
            return {
                "pair_key": pair_key,
                "result_key": result_key,
                "value": sampled_paths,
                "bucket": "all_similar",
                "error": "",
                "refs_total": refs_total,
                "refs_examined": refs_examined,
                "early_stop": early_stop,
            }
        return {
            "pair_key": pair_key,
            "result_key": result_key,
            "value": list(matched_paths),
            "bucket": "matched",
            "error": "",
            "refs_total": refs_total,
            "refs_examined": refs_examined,
            "early_stop": early_stop,
        }

    out_value = decide_matched_paths_output(matched_paths, match_threshold)
    return {
        "pair_key": pair_key,
        "result_key": result_key,
        "value": out_value,
        "bucket": "matched" if out_value else "no_match",
        "error": "",
        "refs_total": refs_total,
        "refs_examined": refs_examined,
        "early_stop": early_stop,
    }


def _worker_queue(model: str, base_url: str, task_queue: mp.Queue, result_queue: mp.Queue, args_obj: Any):
    base.MODEL = model
    base.BASE_URL = base_url
    base.configure_worker_runtime(args_obj)
    global G_ARGS
    G_ARGS = args_obj
    endpoint = f"{model}@{base_url}"
    while True:
        task = task_queue.get()
        if task is None:
            break
        stats_before = base.get_runtime_stats_snapshot()
        task_started_at = time.time()
        try:
            rec = _judge_one(task)
        except Exception as e:
            rec = {
                "pair_key": task.get("pair_key", ""),
                "result_key": task.get("result_key", task.get("pair_key", "")),
                "value": [],
                "bucket": "error",
                "error": f"worker_exception: {e}",
                "refs_total": 0,
                "refs_examined": 0,
                "early_stop": "",
            }
        stats_after = base.get_runtime_stats_snapshot()
        rec["endpoint"] = endpoint
        rec["task_wall_sec"] = max(0.0, time.time() - task_started_at)
        rec["runtime"] = base.diff_runtime_stats(stats_before, stats_after)
        result_queue.put(rec)


def main():
    parser = argparse.ArgumentParser("内容阈值命中判别：支持按 pair 或按图片输出")
    parser.add_argument("--triplet-jsonl", default="")
    parser.add_argument("--content-index-jsonl", default="")
    parser.add_argument("--out-jsonl", default="", help="主输出 jsonl。per-image 模式下写入命中阈值但不是全相似的结果。")
    parser.add_argument("--all-similar-out-jsonl", default="", help="per-image 模式下，全参考图都相似的结果单独写入该 jsonl。")
    parser.add_argument("--error-log-jsonl", default="")
    parser.add_argument("--processed-jsonl", default="", help="可选的断点续跑状态文件，记录所有已处理 key。")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--content_conf_thr", type=float, default=0.5)
    parser.add_argument("--content_judge_times", type=int, default=3)
    parser.add_argument("--content_min_true", type=int, default=3)
    parser.add_argument("--match_threshold", type=int, default=1, help="至少命中多少张content图才视为通过，默认1")
    parser.add_argument(
        "--pair-key-order",
        type=str,
        default="content_style",
        choices=PAIR_KEY_ORDER_CHOICES,
        help="pair_key 中两个 id 的顺序：content_style 表示 content_id__style_id；style_content 表示 style_id__content_id。",
    )
    parser.add_argument("--per-image", action="store_true", help="把输入 jsonl 的 value 列表里的每张图都展开为独立任务，并以图片路径为输出 key。")
    parser.add_argument("--all-similar-sample-size", type=int, default=2, help="per-image 模式下，全相似结果输出时随机采样多少张内容参考图。")
    parser.add_argument(
        "--exact-all-similar",
        action="store_true",
        help="达到 match_threshold 后继续扫描全部参考图，以便精确区分 all_similar；默认关闭，执行真正 first-hit/threshold early-stop。",
    )
    parser.add_argument("--model", type=str, default=base.MODEL)
    parser.add_argument("--base_url", type=str, default=base.BASE_URL)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--procs_per_endpoint", type=int, default=1)
    parser.add_argument("--conn_retry_times", type=int, default=5)
    parser.add_argument("--conn_retry_delay", type=float, default=2.0)
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--image-cache-size", type=int, default=32)
    parser.add_argument("--stats-interval-sec", type=float, default=60.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=32)
    parser.add_argument("--copy-from-jsonl", default="", help="从结果jsonl拷贝 value 中图片路径")
    parser.add_argument("--copy-out-dir", default="", help="拷贝输出目录，支持本地/s3")
    parser.add_argument("--copy-max-items", type=int, default=0, help="最多处理多少条非空记录，<=0 表示全量")
    args = parser.parse_args()
    if int(args.match_threshold) < 0:
        raise RuntimeError("--match_threshold 必须 >= 0")
    if int(args.all_similar_sample_size) <= 0:
        raise RuntimeError("--all_similar_sample_size 必须 > 0")
    if float(args.request_timeout_sec) <= 0:
        raise RuntimeError("--request-timeout-sec 必须 > 0")
    if int(args.image_cache_size) < 0:
        raise RuntimeError("--image-cache-size 必须 >= 0")
    if float(args.stats_interval_sec) <= 0:
        raise RuntimeError("--stats-interval-sec 必须 > 0")
    if int(args.copy_max_items) < 0:
        raise RuntimeError("--copy_max_items 必须 >= 0")

    if args.copy_from_jsonl:
        copy_from_result_jsonl(
            copy_from_jsonl=args.copy_from_jsonl,
            copy_out_dir=args.copy_out_dir,
            copy_max_items=int(args.copy_max_items),
        )
        return

    if not args.triplet_jsonl:
        raise RuntimeError("--triplet-jsonl 不能为空")
    if not args.content_index_jsonl:
        raise RuntimeError("--content-index-jsonl 不能为空")
    if not args.out_jsonl:
        raise RuntimeError("--out-jsonl 不能为空")

    content_index = read_content_index(args.content_index_jsonl)
    tasks, skipped_parse = parse_triplet_jsonl(
        args.triplet_jsonl,
        content_index,
        pair_key_order=str(args.pair_key_order),
        per_image=bool(args.per_image),
    )
    if not tasks:
        raise RuntimeError("没有可处理任务")
    log(f"[Host] pair_key_order={args.pair_key_order}")
    if args.exact_all_similar:
        log("[Host] exact_all_similar=on，将在达到阈值后继续扫描全部参考图")
    else:
        log("[Host] exact_all_similar=off，执行真正 first-hit/threshold early-stop")
    log(
        f"[Host] request_timeout_sec={float(args.request_timeout_sec):.1f} "
        f"image_cache_size={int(args.image_cache_size)} "
        f"stats_interval_sec={float(args.stats_interval_sec):.1f}"
    )

    resume_paths: List[str] = []
    processed_path = args.processed_jsonl.strip()
    requested_all_similar_path = args.all_similar_out_jsonl.strip()
    all_similar_path = requested_all_similar_path if bool(args.exact_all_similar) else ""
    if requested_all_similar_path and not all_similar_path:
        log("[Host] all_similar_out_jsonl 已忽略；当前为真正 first-hit 模式，不再为区分 all_similar 扫完整个参考集")
    if processed_path:
        resume_paths.append(processed_path)
    else:
        resume_paths.append(args.out_jsonl)
        if all_similar_path:
            resume_paths.append(all_similar_path)
    existing_done = load_existing_done(resume_paths) if (not args.overwrite) else {}
    if existing_done:
        before = len(tasks)
        tasks = [t for t in tasks if t.get("result_key") not in existing_done]
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
    log("[Host] candidates:")
    for m, u in endpoints:
        log(f"[Host] candidate model={m} url={u}")
    endpoints = probe_endpoints(endpoints, timeout_sec=max(0.5, float(args.probe_timeout)))
    if not endpoints:
        raise RuntimeError("没有可用endpoint，探测全部失败")
    log(f"[Host] available={len(endpoints)}")

    pp = max(1, int(args.procs_per_endpoint))
    worker_specs: List[Tuple[str, str]] = []
    for _ in range(pp):
        for ep in endpoints:
            worker_specs.append(ep)

    out_path = args.out_jsonl
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if all_similar_path:
        os.makedirs(os.path.dirname(all_similar_path) or ".", exist_ok=True)
    err_path = args.error_log_jsonl.strip()
    if err_path:
        os.makedirs(os.path.dirname(err_path) or ".", exist_ok=True)
    if processed_path:
        os.makedirs(os.path.dirname(processed_path) or ".", exist_ok=True)

    task_queue: mp.Queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)
    for _ in range(len(worker_specs)):
        task_queue.put(None)
    result_queue: mp.Queue = mp.Queue()
    workers: List[mp.Process] = []
    for m, u in worker_specs:
        p = mp.Process(target=_worker_queue, args=(m, u, task_queue, result_queue, args))
        p.daemon = False
        p.start()
        workers.append(p)

    total = len(tasks)
    done = 0
    written = 0
    matched = 0
    all_similar = 0
    no_match = 0
    errs = 0
    flush_every = max(1, int(args.flush_every))
    out_mode = "w" if args.overwrite else "a"
    unit_name = "image" if bool(args.per_image) else "pair"
    endpoint_aggs: Dict[str, Dict[str, float]] = {}
    run_started_at = time.time()
    last_stats_log_at = run_started_at
    last_stats_snapshot: Dict[str, Dict[str, float]] = {}

    def maybe_log_endpoint_stats(force: bool = False):
        nonlocal last_stats_log_at, last_stats_snapshot
        now = time.time()
        if (not force) and (now - last_stats_log_at < float(args.stats_interval_sec)):
            return
        total_elapsed = max(0.001, now - run_started_at)
        window_elapsed = max(0.001, now - last_stats_log_at)
        _log_endpoint_stats(endpoint_aggs, total_elapsed, window_elapsed, last_stats_snapshot)
        last_stats_snapshot = _clone_endpoint_aggs(endpoint_aggs)
        last_stats_log_at = now

    pbar = tqdm(total=total, desc="ContentFirstHit", unit=unit_name)
    with open(out_path, out_mode, encoding="utf-8", buffering=1) as fout:
        fall = open(all_similar_path, out_mode, encoding="utf-8", buffering=1) if all_similar_path else None
        ferr = open(err_path, "w", encoding="utf-8", buffering=1) if err_path else None
        fproc = open(processed_path, out_mode, encoding="utf-8", buffering=1) if processed_path else None
        try:
            while done < total:
                try:
                    rec = result_queue.get(timeout=10.0)
                    done += 1
                except queue.Empty:
                    maybe_log_endpoint_stats(force=False)
                    if any(p.is_alive() for p in workers):
                        continue
                    break
                _update_endpoint_agg(endpoint_aggs, rec)
                result_key = rec.get("result_key", "")
                value = rec.get("value", [])
                bucket = rec.get("bucket", "matched")
                err = rec.get("error", "")
                if bool(args.per_image):
                    if result_key:
                        out_val = value if isinstance(value, list) else []
                        if bucket == "matched":
                            fout.write(json.dumps({result_key: out_val}, ensure_ascii=False) + "\n")
                            written += 1
                            matched += 1
                        elif bucket == "all_similar":
                            target_file = fall if fall is not None else fout
                            target_file.write(json.dumps({result_key: out_val}, ensure_ascii=False) + "\n")
                            written += 1
                            all_similar += 1
                        elif bucket == "no_match":
                            no_match += 1
                        if fproc is not None and bucket != "error":
                            fproc.write(json.dumps({result_key: {"bucket": bucket}}, ensure_ascii=False) + "\n")
                elif result_key:
                    out_val = value if isinstance(value, list) else []
                    fout.write(json.dumps({result_key: out_val}, ensure_ascii=False) + "\n")
                    written += 1
                    if isinstance(out_val, list) and len(out_val) > 0:
                        matched += 1
                    if fproc is not None and bucket != "error":
                        fproc.write(json.dumps({result_key: {"bucket": bucket}}, ensure_ascii=False) + "\n")
                if err and ferr is not None:
                    ferr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    errs += 1
                if done % flush_every == 0:
                    fout.flush()
                    if fall is not None:
                        fall.flush()
                    if ferr is not None:
                        ferr.flush()
                    if fproc is not None:
                        fproc.flush()
                pbar.update(1)
                if done % 200 == 0 or done == total:
                    if bool(args.per_image):
                        ratio = (written / done) if done > 0 else 0.0
                        log(
                            f"progress {done}/{total} kept={written} "
                            f"matched={matched} all_similar={all_similar} no_match={no_match} "
                            f"keep_ratio={ratio:.2%} err={errs}"
                        )
                    else:
                        ratio = (matched / written) if written > 0 else 0.0
                        log(f"progress {done}/{total} written={written} matched={matched} matched_ratio={ratio:.2%} err={errs}")
                    maybe_log_endpoint_stats(force=True)
        finally:
            if fall is not None:
                fall.close()
            if ferr is not None:
                ferr.close()
            if fproc is not None:
                fproc.close()
            pbar.close()

    for p in workers:
        p.join()

    if done < total:
        missing = total - done
        errs += missing
        log(f"[WARN] worker提前退出，未返回结果数量={missing}")
    maybe_log_endpoint_stats(force=True)
    if bool(args.per_image):
        ratio = (written / done) if done > 0 else 0.0
        log(
            f"DONE total={total} done={done} kept={written} matched={matched} "
            f"all_similar={all_similar} no_match={no_match} keep_ratio={ratio:.2%} "
            f"err={errs} skipped_parse={skipped_parse}"
        )
    else:
        ratio = (matched / written) if written > 0 else 0.0
        log(f"DONE total={total} done={done} written={written} matched={matched} matched_ratio={ratio:.2%} err={errs} skipped_parse={skipped_parse}")
    log(f"out_jsonl={out_path}")
    if all_similar_path:
        log(f"all_similar_out_jsonl={all_similar_path}")
    if err_path:
        log(f"error_log_jsonl={err_path}")
    if processed_path:
        log(f"processed_jsonl={processed_path}")


if __name__ == "__main__":
    main()

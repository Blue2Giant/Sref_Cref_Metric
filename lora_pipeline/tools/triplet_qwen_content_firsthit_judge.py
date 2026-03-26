#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import time
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

import triplet_qwen_dual_judge as base


G_ARGS = None


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def parse_triplet_jsonl(path: str, content_index: Dict[str, List[str]]) -> Tuple[List[Dict[str, Any]], int]:
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
                if not main_img:
                    skipped += 1
                    continue
                content_id, _sid = pair_key.split("__", 1)
                content_id = content_id.strip()
                content_imgs = content_index.get(content_id, [])
                tasks.append(
                    {
                        "pair_key": pair_key,
                        "main_img": main_img,
                        "content_id": content_id,
                        "content_imgs": content_imgs,
                    }
                )
    return tasks, skipped


def load_existing_done(out_jsonl: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not out_jsonl or not os.path.isfile(out_jsonl):
        return out
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
                if not isinstance(k, str):
                    continue
                if isinstance(v, list):
                    out[k] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, str):
                    out[k] = v
    return out


def decide_matched_paths_output(matched_paths: List[str], match_threshold: int) -> List[str]:
    if int(match_threshold) < 0:
        raise ValueError("match_threshold 必须 >= 0")
    if len(matched_paths) >= int(match_threshold):
        return list(matched_paths[: int(match_threshold)])
    return []


def _judge_one(task: Dict[str, Any]) -> Dict[str, Any]:
    args = G_ARGS
    pair_key = task["pair_key"]
    main_img = task["main_img"]
    content_imgs = task["content_imgs"]

    if not base.smart_exists(main_img):
        return {"pair_key": pair_key, "value": [], "error": f"main_not_found: {main_img}"}
    if not content_imgs:
        return {"pair_key": pair_key, "value": [], "error": f"content_id_not_found: {task['content_id']}"}

    matched_paths: List[str] = []
    for cp in content_imgs:
        if not base.smart_exists(cp):
            continue
        decision, _detail, retry_exhausted = base.judge_pair_voting(
            path_a=main_img,
            path_b=cp,
            system_prompt=base.CONTENT_SYSTEM_PROMPT,
            user_instruction=base.CONTENT_USER_INSTRUCTION,
            conf_thr=float(args.content_conf_thr),
            judge_times=int(args.content_judge_times),
            min_true=int(args.content_min_true),
        )
        if retry_exhausted:
            return {"pair_key": pair_key, "value": [], "error": "retry_exhausted"}
        if decision is True:
            matched_paths.append(cp)

    out_value = decide_matched_paths_output(matched_paths, int(args.match_threshold))
    return {"pair_key": pair_key, "value": out_value, "error": ""}


def _worker(model: str, base_url: str, tasks: List[Dict[str, Any]], result_queue: mp.Queue, args_obj: Any):
    base.MODEL = model
    base.BASE_URL = base_url
    base.G_ARGS = args_obj
    global G_ARGS
    G_ARGS = args_obj
    for task in tasks:
        try:
            result_queue.put(_judge_one(task))
        except Exception as e:
            result_queue.put({"pair_key": task.get("pair_key", ""), "value": [], "error": f"worker_exception: {e}"})


def main():
    parser = argparse.ArgumentParser("内容阈值命中判别：命中数量达到阈值才输出content路径列表")
    parser.add_argument("--triplet-jsonl", default="")
    parser.add_argument("--content-index-jsonl", default="")
    parser.add_argument("--out-jsonl", default="")
    parser.add_argument("--error-log-jsonl", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--content_conf_thr", type=float, default=0.5)
    parser.add_argument("--content_judge_times", type=int, default=3)
    parser.add_argument("--content_min_true", type=int, default=3)
    parser.add_argument("--match_threshold", type=int, default=1, help="至少命中多少张content图才视为通过，默认1")
    parser.add_argument("--model", type=str, default=base.MODEL)
    parser.add_argument("--base_url", type=str, default=base.BASE_URL)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--procs_per_endpoint", type=int, default=1)
    parser.add_argument("--conn_retry_times", type=int, default=5)
    parser.add_argument("--conn_retry_delay", type=float, default=2.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--copy-from-jsonl", default="", help="从结果jsonl拷贝 value 中图片路径")
    parser.add_argument("--copy-out-dir", default="", help="拷贝输出目录，支持本地/s3")
    parser.add_argument("--copy-max-items", type=int, default=0, help="最多处理多少条非空记录，<=0 表示全量")
    args = parser.parse_args()
    if int(args.match_threshold) < 0:
        raise RuntimeError("--match_threshold 必须 >= 0")
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
    tasks, skipped_parse = parse_triplet_jsonl(args.triplet_jsonl, content_index)
    if not tasks:
        raise RuntimeError("没有可处理任务")

    existing_done = load_existing_done(args.out_jsonl) if (not args.overwrite) else {}
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
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(len(worker_specs))]
    for idx, t in enumerate(tasks):
        chunks[idx % len(worker_specs)].append(t)

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
        p = mp.Process(target=_worker, args=(m, u, sub, result_queue, args))
        p.daemon = False
        p.start()
        workers.append(p)

    total = len(tasks)
    done = 0
    ok = 0
    matched = 0
    errs = 0
    flush_every = max(1, int(args.flush_every))
    out_mode = "w" if args.overwrite else "a"
    pbar = tqdm(total=total, desc="ContentFirstHit", unit="pair")
    with open(out_path, out_mode, encoding="utf-8", buffering=1) as fout:
        ferr = open(err_path, "w", encoding="utf-8", buffering=1) if err_path else None
        try:
            while done < total:
                try:
                    rec = result_queue.get(timeout=10.0)
                    done += 1
                except queue.Empty:
                    if any(p.is_alive() for p in workers):
                        continue
                    break
                pair_key = rec.get("pair_key", "")
                value = rec.get("value", [])
                err = rec.get("error", "")
                if pair_key:
                    out_val = value if isinstance(value, list) else []
                    fout.write(json.dumps({pair_key: out_val}, ensure_ascii=False) + "\n")
                    ok += 1
                    if isinstance(out_val, list) and len(out_val) > 0:
                        matched += 1
                if err and ferr is not None:
                    ferr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    errs += 1
                if done % flush_every == 0:
                    fout.flush()
                    if ferr is not None:
                        ferr.flush()
                pbar.update(1)
                if done % 200 == 0 or done == total:
                    ratio = (matched / ok) if ok > 0 else 0.0
                    log(f"progress {done}/{total} written={ok} matched={matched} matched_ratio={ratio:.2%} err={errs}")
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
    ratio = (matched / ok) if ok > 0 else 0.0
    log(f"DONE total={total} done={done} written={ok} matched={matched} matched_ratio={ratio:.2%} err={errs} skipped_parse={skipped_parse}")
    log(f"out_jsonl={out_path}")
    if err_path:
        log(f"error_log_jsonl={err_path}")


if __name__ == "__main__":
    main()

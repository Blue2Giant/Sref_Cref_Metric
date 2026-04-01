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

import triplet_qwen_style_index_judge as base


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


def read_style_index(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not os.path.isfile(path):
        raise RuntimeError(f"style索引文件不存在: {path}")
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


def parse_triplet_jsonl(path: str, style_index: Dict[str, List[str]], per_image: bool = False) -> Tuple[List[Dict[str, Any]], int]:
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
                _cid, style_id = pair_key.split("__", 1)
                style_id = style_id.strip()
                style_imgs = style_index.get(style_id, [])
                if per_image:
                    main_imgs = _dedupe_preserve_order([x for x in arr if isinstance(x, str)])
                else:
                    main_img = str(arr[0]).strip()
                    main_imgs = [main_img] if main_img else []
                if not main_imgs:
                    skipped += 1
                    continue
                for main_img in main_imgs:
                    tasks.append(
                        {
                            "pair_key": pair_key,
                            "result_key": main_img if per_image else pair_key,
                            "main_img": main_img,
                            "style_id": style_id,
                            "style_imgs": style_imgs,
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


def _judge_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """返回每个任务的判别输出。per-image 模式下按 bucket 分流到不同 jsonl。"""
    args = G_ARGS
    pair_key = task["pair_key"]
    result_key = task.get("result_key", pair_key)
    main_img = task["main_img"]
    style_imgs = task["style_imgs"]

    if not base.smart_exists(main_img):
        return {
            "pair_key": pair_key,
            "result_key": result_key,
            "value": [],
            "bucket": "error",
            "error": f"main_not_found: {main_img}",
        }
    if not style_imgs:
        return {
            "pair_key": pair_key,
            "result_key": result_key,
            "value": [],
            "bucket": "error",
            "error": f"style_id_not_found: {task['style_id']}",
        }

    existing_style_imgs = _dedupe_preserve_order([sp for sp in style_imgs if base.smart_exists(sp)])
    if not existing_style_imgs:
        return {
            "pair_key": pair_key,
            "result_key": result_key,
            "value": [],
            "bucket": "error",
            "error": f"style_refs_not_found: {task['style_id']}",
        }

    matched_paths: List[str] = []
    for sp in existing_style_imgs:
        decision, _detail, retry_exhausted = base.judge_pair_voting(
            path_a=main_img,
            path_b=sp,
            system_prompt=base.STYLE_SYSTEM_PROMPT,
            user_instruction=base.STYLE_USER_INSTRUCTION,
            conf_thr=float(args.style_conf_thr),
            judge_times=int(args.style_judge_times),
            min_true=int(args.style_min_true),
        )
        if retry_exhausted:
            return {
                "pair_key": pair_key,
                "result_key": result_key,
                "value": [],
                "bucket": "error",
                "error": "retry_exhausted",
            }
        if decision is True:
            matched_paths.append(sp)

    if bool(getattr(args, "per_image", False)):
        if (not matched_paths) or (len(matched_paths) < int(args.match_threshold)):
            return {
                "pair_key": pair_key,
                "result_key": result_key,
                "value": [],
                "bucket": "no_match",
                "error": "",
            }
        if len(matched_paths) == len(existing_style_imgs):
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
            }
        return {
            "pair_key": pair_key,
            "result_key": result_key,
            "value": list(matched_paths),
            "bucket": "matched",
            "error": "",
        }

    # 兼容旧行为：按 pair 输出，命中阈值才写入固定数量的 style 路径，否则写空列表。
    out_value = decide_matched_paths_output(matched_paths, int(args.match_threshold))
    return {
        "pair_key": pair_key,
        "result_key": result_key,
        "value": out_value,
        "bucket": "matched" if out_value else "no_match",
        "error": "",
    }


def _worker_queue(model: str, base_url: str, task_queue: mp.Queue, result_queue: mp.Queue, args_obj: Any):
    base.MODEL = model
    base.BASE_URL = base_url
    base.G_ARGS = args_obj
    global G_ARGS
    G_ARGS = args_obj
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            result_queue.put(_judge_one(task))
        except Exception as e:
            result_queue.put(
                {
                    "pair_key": task.get("pair_key", ""),
                    "result_key": task.get("result_key", task.get("pair_key", "")),
                    "value": [],
                    "bucket": "error",
                    "error": f"worker_exception: {e}",
                }
            )


def main():
    parser = argparse.ArgumentParser("风格阈值命中判别：支持按 pair 或按图片输出")
    parser.add_argument("--triplet-jsonl", required=True)
    parser.add_argument("--style-index-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True, help="主输出 jsonl。per-image 模式下写入命中阈值但不是全相似的结果。")
    parser.add_argument("--all-similar-out-jsonl", default="", help="per-image 模式下，全参考图都相似的结果单独写入该 jsonl。")
    parser.add_argument("--error-log-jsonl", default="")
    parser.add_argument("--processed-jsonl", default="", help="可选的断点续跑状态文件，记录所有已处理 key。")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--style_conf_thr", type=float, default=0.5)
    parser.add_argument("--style_judge_times", type=int, default=3)
    parser.add_argument("--style_min_true", type=int, default=3)
    parser.add_argument("--match_threshold", type=int, default=1, help="至少命中多少张style图才视为通过，默认1")
    parser.add_argument("--per-image", action="store_true", help="把输入 jsonl 的 value 列表里的每张图都展开为独立任务，并以图片路径为输出 key。")
    parser.add_argument("--all-similar-sample-size", type=int, default=2, help="per-image 模式下，全相似结果输出时随机采样多少张风格参考图。")
    parser.add_argument("--model", type=str, default=base.MODEL)
    parser.add_argument("--base_url", type=str, default=base.BASE_URL)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--procs_per_endpoint", type=int, default=1)
    parser.add_argument("--conn_retry_times", type=int, default=5)
    parser.add_argument("--conn_retry_delay", type=float, default=2.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    args = parser.parse_args()
    if int(args.match_threshold) < 0:
        raise RuntimeError("--match_threshold 必须 >= 0")
    if int(args.all_similar_sample_size) <= 0:
        raise RuntimeError("--all_similar_sample_size 必须 > 0")

    style_index = read_style_index(args.style_index_jsonl)
    tasks, skipped_parse = parse_triplet_jsonl(args.triplet_jsonl, style_index, per_image=bool(args.per_image))
    if not tasks:
        raise RuntimeError("没有可处理任务")

    resume_paths: List[str] = []
    processed_path = args.processed_jsonl.strip()
    all_similar_path = args.all_similar_out_jsonl.strip()
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
    pbar = tqdm(total=total, desc="StyleFirstHit", unit=unit_name)
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
                    if any(p.is_alive() for p in workers):
                        continue
                    break
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

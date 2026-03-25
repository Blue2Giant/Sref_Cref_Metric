#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import triplet_qwen_dual_judge as dual


G_TRIPLET_DIR = ""
G_STYLE_DIRS: List[str] = []
G_ARGS = None


def _process_one_name_style(name: str) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    triplet_dir = G_TRIPLET_DIR
    style_dirs = G_STYLE_DIRS
    args = G_ARGS
    main_img_path = dual.join_path(triplet_dir, name)

    passed_style = 0
    total_style = len(style_dirs)
    per_style_details = []

    for sdir in style_dirs:
        style_path = dual.join_path(sdir, name)
        style_tag = os.path.basename(sdir.rstrip("/"))
        if not dual.smart_exists(style_path):
            per_style_details.append({"dir": style_tag, "exists": False})
            continue

        if args.style_repeat_only_style1 and style_tag != "style_1":
            pred, reason, conf = dual.direct_judge_images_generic(
                main_img_path,
                style_path,
                dual.STYLE_SYSTEM_PROMPT,
                dual.STYLE_USER_INSTRUCTION,
            )
            if pred is None and reason == dual.RETRY_EXHAUSTED_REASON:
                return None, 0, True
            decision = bool(pred is True and conf is not None and conf > args.style_conf_thr)
            detail = {"status": "single", "pred": pred, "conf": conf, "reason": reason}
        else:
            decision, detail, retry_exhausted = dual.judge_pair_voting(
                path_a=main_img_path,
                path_b=style_path,
                system_prompt=dual.STYLE_SYSTEM_PROMPT,
                user_instruction=dual.STYLE_USER_INSTRUCTION,
                conf_thr=args.style_conf_thr,
                judge_times=args.style_judge_times,
                min_true=args.style_min_true,
            )
            if retry_exhausted:
                return None, 0, True
            decision = bool(decision is True)

        if decision:
            passed_style += 1
        per_style_details.append({"dir": style_tag, "exists": True, "decision": decision, "detail": detail})

    style_r = passed_style / float(total_style) if total_style > 0 else 0.0
    style_pass = style_r >= args.style_ratio
    rec = {
        "name": name,
        "main_img": main_img_path,
        "style_pass": style_pass,
        "style_ratio": style_r,
        "style_passed_cnt": passed_style,
        "style_total": total_style,
        "style_details": per_style_details,
    }
    return rec, 1 if style_pass else 0, False


def _process_one_record_style(task: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    args = G_ARGS
    main_img_path = task.get("main_img", "")
    name = task.get("name", os.path.basename(main_img_path))
    style_items: List[Tuple[str, str]] = task.get("style_items", [])

    passed_style = 0
    total_style = len(style_items)
    per_style_details = []

    for style_tag, style_path in style_items:
        if not dual.smart_exists(style_path):
            per_style_details.append({"dir": style_tag, "exists": False})
            continue

        if args.style_repeat_only_style1 and style_tag != "style_1":
            pred, reason, conf = dual.direct_judge_images_generic(
                main_img_path,
                style_path,
                dual.STYLE_SYSTEM_PROMPT,
                dual.STYLE_USER_INSTRUCTION,
            )
            if pred is None and reason == dual.RETRY_EXHAUSTED_REASON:
                return None, 0, True
            decision = bool(pred is True and conf is not None and conf > args.style_conf_thr)
            detail = {"status": "single", "pred": pred, "conf": conf, "reason": reason}
        else:
            decision, detail, retry_exhausted = dual.judge_pair_voting(
                path_a=main_img_path,
                path_b=style_path,
                system_prompt=dual.STYLE_SYSTEM_PROMPT,
                user_instruction=dual.STYLE_USER_INSTRUCTION,
                conf_thr=args.style_conf_thr,
                judge_times=args.style_judge_times,
                min_true=args.style_min_true,
            )
            if retry_exhausted:
                return None, 0, True
            decision = bool(decision is True)

        if decision:
            passed_style += 1
        per_style_details.append({"dir": style_tag, "exists": True, "decision": decision, "detail": detail})

    style_r = passed_style / float(total_style) if total_style > 0 else 0.0
    style_pass = style_r >= args.style_ratio
    rec = {
        "name": name,
        "main_img": main_img_path,
        "style_pass": style_pass,
        "style_ratio": style_r,
        "style_passed_cnt": passed_style,
        "style_total": total_style,
        "style_details": per_style_details,
    }
    return rec, 1 if style_pass else 0, False


def _worker_process_main_style(model: str, base_url: str, tasks: List[str], result_queue: mp.Queue):
    dual.MODEL = model
    dual.BASE_URL = base_url
    for name in tasks:
        rec, ok_inc, skipped = _process_one_name_style(name)
        result_queue.put((rec, ok_inc, skipped))


def _worker_process_main_records_style(model: str, base_url: str, tasks: List[Dict[str, Any]], result_queue: mp.Queue):
    dual.MODEL = model
    dual.BASE_URL = base_url
    for task in tasks:
        rec, ok_inc, skipped = _process_one_record_style(task)
        result_queue.put((rec, ok_inc, skipped))


def main():
    global G_TRIPLET_DIR, G_STYLE_DIRS, G_ARGS

    ap = argparse.ArgumentParser("仅评估画风相似度")
    ap.add_argument("--root", default="", help="输出目录根：包含 style_and_content/ style_*/")
    ap.add_argument("--input_jsonl", default="", help="jsonl 输入，每行包含 style_and_content/style_* 的图片路径")
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--style_conf_thr", type=float, default=0.5)
    ap.add_argument("--style_judge_times", type=int, default=3)
    ap.add_argument("--style_min_true", type=int, default=2)
    ap.add_argument("--style_ratio", type=float, default=0.66)
    ap.add_argument("--style_repeat_only_style1", action="store_true")
    ap.add_argument("--style_id_txt", default="")

    ap.add_argument("--model", type=str, default=dual.MODEL)
    ap.add_argument("--base_url", type=str, default=dual.BASE_URL)
    ap.add_argument("--endpoint", action="append", default=[])
    ap.add_argument("--procs_per_endpoint", type=int, default=0)
    ap.add_argument("--conn_retry_times", type=int, default=5)
    ap.add_argument("--conn_retry_delay", type=float, default=2.0)

    ap.add_argument("--out_all", required=True, help="全量 map json (Style Pass)")
    ap.add_argument("--out_pos", required=True, help="正样本 map json (Style Pass)")
    ap.add_argument("--out_neg", required=True, help="负样本 map json (Style Pass)")
    ap.add_argument("--out_detail", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num_procs", type=int, default=0)
    args = ap.parse_args()

    existing_all_map = {}
    processed_keys = set()
    if (not args.overwrite) and dual.smart_exists(args.out_all):
        tmp = dual.smart_read_json(args.out_all)
        if isinstance(tmp, dict):
            existing_all_map = tmp
            processed_keys = set(tmp.keys())
            dual.log(f"[Resume] 从已有 out_all 读取到 {len(processed_keys)} 条结果")

    endpoints: List[Tuple[str, str]] = []
    if args.endpoint:
        for e in args.endpoint:
            s = str(e).strip()
            if not s:
                continue
            if "@" in s:
                m_name, url = s.split("@", 1)
                m_name = m_name.strip()
                url = url.strip()
            else:
                m_name = args.model
                url = s
            if m_name and url:
                endpoints.append((m_name, url))
    if not endpoints:
        dual.MODEL = args.model
        dual.BASE_URL = args.base_url

    use_jsonl = bool(args.input_jsonl)
    picked = []
    tasks: List[Dict[str, Any]] = []
    cand_names: List[str] = []

    if use_jsonl:
        if not dual.smart_exists(args.input_jsonl):
            raise RuntimeError(f"input_jsonl 不存在: {args.input_jsonl}")
        with dual.mopen(args.input_jsonl, "r", encoding="utf-8") as f:
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
                main_img = obj.get("style_and_content")
                if not isinstance(main_img, str) or not main_img:
                    continue
                content_id, style_id = dual._extract_pair_ids_from_path(main_img)
                base_name = os.path.basename(main_img)
                if content_id and style_id:
                    name = f"{style_id}__{content_id}_{base_name}"
                else:
                    name = base_name
                tasks.append(
                    {
                        "name": name,
                        "main_img": main_img,
                        "style_items": dual._collect_prefixed_items(obj, "style_"),
                    }
                )
        if not tasks:
            raise RuntimeError(f"{args.input_jsonl} 中未解析到有效样本")
        import random

        rng = random.Random(args.seed)
        picked = list(tasks) if args.num_samples <= 0 or args.num_samples >= len(tasks) else rng.sample(tasks, args.num_samples)
        if processed_keys:
            picked = [task for task in picked if os.path.splitext(task["name"])[0] not in processed_keys]
            if not picked:
                dual.log("[Resume] 没有新的样本需要判别，直接退出。")
                return
    else:
        if not args.root:
            raise RuntimeError("--root 不能为空")
        triplet_dir, _content_dirs_unused = dual.list_dirs(args.root, "content")
        _, style_dirs = dual.list_dirs(args.root, "style")
        if not style_dirs:
            raise RuntimeError(f"在 {args.root} 下没找到 style_*/ 目录")
        cand_names = dual.list_candidate_names(triplet_dir)
        if not cand_names:
            raise RuntimeError(f"{triplet_dir} 下没找到图片")
        style_ids = dual.read_id_txt(args.style_id_txt)
        style_id_set = set(style_ids)
        if style_id_set:
            filtered = []
            for name in cand_names:
                sid, _cid = dual.extract_content_style_ids(name)
                if sid in style_id_set:
                    filtered.append(name)
            cand_names = filtered
            if not cand_names:
                dual.log("[Filter] 过滤后没有候选图片，直接退出。")
                return
        import random

        rng = random.Random(args.seed)
        picked = list(cand_names) if args.num_samples <= 0 or args.num_samples >= len(cand_names) else rng.sample(cand_names, args.num_samples)
        if processed_keys:
            picked = [name for name in picked if os.path.splitext(name)[0] not in processed_keys]
            if not picked:
                dual.log("[Resume] 没有新的样本需要判别，直接退出。")
                return
        G_TRIPLET_DIR = triplet_dir
        G_STYLE_DIRS = style_dirs
    G_ARGS = args
    dual.G_ARGS = args

    results: List[Dict[str, Any]] = []
    style_ok_cnt = 0
    skipped_cnt = 0

    all_map: Dict[str, int] = dict(existing_all_map)
    pos_map: Dict[str, int] = {}
    neg_map: Dict[str, int] = {}
    if (not args.overwrite) and dual.smart_exists(args.out_pos):
        tmp_pos = dual.smart_read_json(args.out_pos)
        if isinstance(tmp_pos, dict):
            pos_map = dict(tmp_pos)
    if (not args.overwrite) and dual.smart_exists(args.out_neg):
        tmp_neg = dual.smart_read_json(args.out_neg)
        if isinstance(tmp_neg, dict):
            neg_map = dict(tmp_neg)

    def _update_maps_and_flush(rec: Dict[str, Any]):
        nonlocal style_ok_cnt
        base_key = os.path.splitext(rec["name"])[0]
        v = 1 if rec["style_pass"] else 0
        all_map[base_key] = v
        if v == 1:
            pos_map[base_key] = 1
        else:
            neg_map[base_key] = 0
        if rec["style_pass"]:
            style_ok_cnt += 1
        dual.smart_write_json(args.out_all, all_map)
        dual.smart_write_json(args.out_pos, pos_map)
        dual.smart_write_json(args.out_neg, neg_map)

    if endpoints:
        per = args.procs_per_endpoint if args.procs_per_endpoint and args.procs_per_endpoint > 0 else 1
        workers: List[mp.Process] = []
        result_queue: mp.Queue = mp.Queue()
        worker_specs: List[Tuple[str, str]] = []
        for _ in range(per):
            for model_name, url in endpoints:
                worker_specs.append((model_name, url))
        worker_count = len(worker_specs)
        sliced: List[List[Any]] = [[] for _ in range(worker_count)]
        for idx, item in enumerate(picked):
            sliced[idx % worker_count].append(item)
        for i, (model_name, url) in enumerate(worker_specs):
            sub_tasks = sliced[i]
            if not sub_tasks:
                continue
            if use_jsonl:
                p = mp.Process(target=_worker_process_main_records_style, args=(model_name, url, sub_tasks, result_queue))
            else:
                p = mp.Process(target=_worker_process_main_style, args=(model_name, url, sub_tasks, result_queue))
            p.daemon = False
            p.start()
            workers.append(p)
        total = len(picked)
        for _ in tqdm(range(total), desc="StyleJudge-MP", unit="img"):
            rec, ok_inc, skipped = result_queue.get()
            if skipped or rec is None:
                skipped_cnt += 1
                continue
            results.append(rec)
            _update_maps_and_flush(rec)
        for p in workers:
            p.join()
    elif args.num_procs and args.num_procs > 1:
        procs = max(1, int(args.num_procs))
        with mp.Pool(processes=procs) as pool:
            it = pool.imap_unordered(_process_one_record_style, picked) if use_jsonl else pool.imap_unordered(_process_one_name_style, picked)
            for rec, _ok_inc, skipped in tqdm(it, total=len(picked), desc="StyleJudge-MP", unit="img"):
                if skipped or rec is None:
                    skipped_cnt += 1
                    continue
                results.append(rec)
                _update_maps_and_flush(rec)
    else:
        for i, item in enumerate(picked, 1):
            rec, _ok_inc, skipped = _process_one_record_style(item) if use_jsonl else _process_one_name_style(item)
            if skipped or rec is None:
                skipped_cnt += 1
                continue
            results.append(rec)
            dual.log(f"[{i}/{len(picked)}] {rec['name']} -> Style:{rec['style_pass']} ({rec['style_ratio']:.2f})")
            _update_maps_and_flush(rec)

    processed_cnt = max(0, len(picked) - skipped_cnt)
    processed_den = processed_cnt if processed_cnt > 0 else 1
    dual.log(f"[DONE] Processed {len(picked)} samples.")
    dual.log(f"Style Pass: {style_ok_cnt} ({style_ok_cnt/processed_den:.2%})")
    dual.log(f"Skipped: {skipped_cnt}")
    dual.log(f"  -> {args.out_all}")
    dual.log(f"  -> {args.out_pos}")
    dual.log(f"  -> {args.out_neg}")

    if args.out_detail:
        summary = {
            "root": args.root,
            "picked": len(picked),
            "processed": processed_cnt,
            "style_ok": style_ok_cnt,
            "args": vars(args),
            "skipped": skipped_cnt,
        }
        dual.smart_write_json(args.out_detail, {"summary": summary, "results": results})
        dual.log(f"[Detail] -> {args.out_detail}")


if __name__ == "__main__":
    main()

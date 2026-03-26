#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Qwen style-lp 多进程推理启动器")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--run_py", default="/data/benchmark_metrics/insight/qwen_2511_style_lp_guided_demo.py")
    p.add_argument("--prompts_json", required=True)
    p.add_argument("--cref_dir", required=True)
    p.add_argument("--sref_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--key_txt", default="", help="可选，留空时自动使用prompts_json中的全部key")
    p.add_argument("--gpu_groups", required=True, help='用分号分组，如 "0,1;2,3"')
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--true_cfg_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment", default="lp_restore")
    p.add_argument("--lp_factor", type=int, default=4)
    p.add_argument("--beta_schedule", default="piecewise")
    p.add_argument("--early_ratio", type=float, default=0.35)
    p.add_argument("--beta_early", type=float, default=0.0)
    p.add_argument("--beta_mid", type=float, default=0.2)
    p.add_argument("--beta_late", type=float, default=0.5)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--attention_slicing", default="max")
    p.add_argument("--device_map", default="balanced")
    p.add_argument("--max_memory_gpu", default="70GiB,70GiB")
    p.add_argument("--max_memory_cpu", default="800GiB")
    p.add_argument("--empty_cache_per_step", type=int, default=4)
    return p.parse_args()


def read_keys(path: Path):
    keys = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if s and s not in seen:
                keys.append(s)
                seen.add(s)
    if not keys:
        raise RuntimeError(f"key_txt为空: {path}")
    return keys


def read_keys_from_prompts(prompts_json: Path):
    with prompts_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError(f"prompts_json无有效key: {prompts_json}")
    return sorted(str(k) for k in data.keys())


def write_keys(path: Path, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k in keys:
            f.write(f"{k}\n")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_shard_dir = out_dir / "_key_shards"
    log_dir = out_dir / "_logs"
    key_shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_groups = [g.strip() for g in str(args.gpu_groups).split(";") if g.strip()]
    if len(gpu_groups) <= 1:
        raise RuntimeError("gpu_groups 至少要有两组，例如 0,1;2,3")
    nproc = len(gpu_groups)

    key_txt = str(args.key_txt).strip()
    if key_txt and Path(key_txt).exists():
        keys = read_keys(Path(key_txt))
    else:
        keys = read_keys_from_prompts(Path(args.prompts_json))
        print(f"[INFO] key_txt为空或不存在，使用prompts_json全量key，总数={len(keys)}")
    shards = [[] for _ in range(nproc)]
    for i, k in enumerate(keys):
        shards[i % nproc].append(k)

    procs = []
    for i in range(nproc):
        shard_keys = shards[i]
        if len(shard_keys) == 0:
            continue
        shard_key_txt = key_shard_dir / f"worker_{i}.txt"
        write_keys(shard_key_txt, shard_keys)
        worker_metrics = out_dir / f"metrics.worker{i}.jsonl"
        worker_log = log_dir / f"worker_{i}.log"
        gpu_group = gpu_groups[i]
        local_gpu_count = len([x for x in gpu_group.split(",") if x.strip()])
        local_gpus_arg = ",".join(str(x) for x in range(local_gpu_count))

        cmd = [
            args.python_bin,
            args.run_py,
            "--prompts_json",
            args.prompts_json,
            "--cref_dir",
            args.cref_dir,
            "--sref_dir",
            args.sref_dir,
            "--out_dir",
            str(out_dir),
            "--model_name",
            args.model_name,
            "--gpus",
            local_gpus_arg,
            "--key_txt",
            str(shard_key_txt),
            "--steps",
            str(args.steps),
            "--true-cfg-scale",
            str(args.true_cfg_scale),
            "--seed",
            str(args.seed + i * 10000),
            "--experiment",
            args.experiment,
            "--lp-factor",
            str(args.lp_factor),
            "--beta-schedule",
            args.beta_schedule,
            "--early-ratio",
            str(args.early_ratio),
            "--beta-early",
            str(args.beta_early),
            "--beta-mid",
            str(args.beta_mid),
            "--beta-late",
            str(args.beta_late),
            "--max-sequence-length",
            str(args.max_sequence_length),
            "--attention-slicing",
            args.attention_slicing,
            "--device-map",
            args.device_map,
            "--max-memory-gpu",
            args.max_memory_gpu,
            "--max-memory-cpu",
            args.max_memory_cpu,
            "--enable-vae-slicing",
            "--enable-vae-tiling",
            "--offload-image-latents-to-cpu",
            "--offload-prompt-embeds-to-cpu",
            "--empty-cache-per-step",
            str(args.empty_cache_per_step),
            "--metrics_jsonl",
            str(worker_metrics),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_group
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        with worker_log.open("w", encoding="utf-8") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append((i, p, worker_log))
        print(f"[LAUNCH] worker={i} gpus={gpu_group} keys={len(shard_keys)} log={worker_log}")

    failed = []
    for i, p, worker_log in procs:
        ret = p.wait()
        if ret != 0:
            failed.append((i, ret, worker_log))
        print(f"[EXIT] worker={i} code={ret}")

    merged_metrics = out_dir / "metrics.jsonl"
    with merged_metrics.open("w", encoding="utf-8") as wf:
        for i in range(nproc):
            src = out_dir / f"metrics.worker{i}.jsonl"
            if src.exists():
                wf.write(src.read_text(encoding="utf-8"))
    print(f"[MERGE] metrics={merged_metrics}")

    if failed:
        msg = "; ".join([f"worker={i},code={c},log={l}" for i, c, l in failed])
        raise SystemExit(f"多进程推理失败: {msg}")
    print("[DONE] all workers finished")


if __name__ == "__main__":
    main()

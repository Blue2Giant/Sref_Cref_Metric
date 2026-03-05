#!/usr/bin/env bash
set -euo pipefail

sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content

python /data/benchmark_metrics/CSGO/run_csgo.py \
  --prompts_json $sref_root/prompts.json \
  --cref_dir $sref_root/cref \
  --sref_dir $sref_root/sref \
  --out_dir $sref_root/csgo \
  --steps 50 \
  --guidance_scale 10.0 \
  --content_scale 1.0 \
  --style_scale 1.0 \
  --controlnet_conditioning_scale 0.4 \
  --gpus 0,1,2,3 \
  --save_jsonl

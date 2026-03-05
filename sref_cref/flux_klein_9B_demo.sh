#!/usr/bin/env bash
set -euo pipefail

sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content

python /data/benchmark_metrics/sref_cref/flux_klein_9B.py \
  --prompts_json $sref_root/prompts.json \
  --cref_dir $sref_root/cref \
  --sref_dir $sref_root/sref \
  --out_dir $sref_root/flux-klein-9b \
  --model_name /mnt/jfs/model_zoo/FLUX.2-klein-9B/ \
  --steps 4 \
  --guidance_scale 1.0 \
  --gpus 0 \
  --input_resolution 1024x1024 \
  --save_jsonl \
  --overwrite 
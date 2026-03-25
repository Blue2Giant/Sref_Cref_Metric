#!/usr/bin/env bash
python /data/benchmark_metrics/insight/qwen_2511_style_lp_guided_demo.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref \
  --out_dir /data/benchmark_metrics/logs/qwen-edit-style-lp \
  --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
  --gpus 0 \
  --key_txt /data/benchmark_metrics/insight/key.txt \
  --steps 28 \
  --true-cfg-scale 4.0 \
  --experiment lp_restore \
  --lp-factor 4 \
  --beta-schedule piecewise \
  --early-ratio 0.35 \
  --beta-early 0.0 \
  --beta-mid 0.2 \
  --beta-late 0.5 \
  --metrics_jsonl /data/benchmark_metrics/logs/qwen-edit-style-lp/metrics.jsonl

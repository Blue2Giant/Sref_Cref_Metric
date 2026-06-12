#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-/data/Miniconda/.conda/envs/diffsynth/bin/python}

sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
"$PYTHON_BIN" /data/benchmark_metrics/sref_cref/qwen_infer.py \
    --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/prompts.json \
    --cref_dir $sref_root/cref \
    --sref_dir $sref_root/sref \
    --out_dir $sref_root/qwen-edit_1024x1024 \
    --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
    --output_resolution 1024x1024 \
    --gpus 0

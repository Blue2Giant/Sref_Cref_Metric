#!/usr/bin/env bash
overwrite=${OVERWRITE:-0}
gpu_ids=${1:-${GPU_IDS:-0}}
extra_args=""
if [ "$overwrite" = "1" ]; then
  extra_args="--overwrite"
fi

python /data/benchmark_metrics/insight/qwen_2511_attention_fullmap.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref \
  --out_dir /mnt/jfs/qwen-edit-attn-fullmap-keycolor \
  --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
  --gpus "$gpu_ids" \
  --key_txt /data/benchmark_metrics/insight/analysis_key.txt \
  --steps 28 \
  --true-cfg-scale 4.0 \
  --max-tokens 128 \
  --max-q-tokens 128 \
  --max-k-tokens 192 \
  --aggregate-head mean \
  --aggregate-block mean \
  --step-stride 4 \
  --block-stride 4 \
  --panel-size 1.1 \
  --image-dpi 240 \
  --ref-labels cref,sref \
  --attn-cmap turbo \
  --attn-gamma 0.48 \
  --high-attn-quantile 0.9 \
  --high-attn-contour-color "#00e5ff" \
  --high-attn-contour-width 1.0 \
  --boundary-linewidth 1.6 \
  --save-format png \
  $extra_args

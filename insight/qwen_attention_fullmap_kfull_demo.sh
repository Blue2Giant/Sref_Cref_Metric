#!/usr/bin/env bash
overwrite=${OVERWRITE:-0}
gpu_ids="${1:-${GPU_IDS:-}}"
extra_args=""
if [ "$overwrite" = "1" ]; then
  extra_args="--overwrite"
fi

if [ -z "$gpu_ids" ]; then
  detected_gpu_count="$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)"
  if [ "${detected_gpu_count:-0}" -gt 0 ]; then
    gpu_ids="$(python -c "import torch; print(','.join(str(i) for i in range(torch.cuda.device_count())))" 2>/dev/null)"
  else
    gpu_ids="0"
  fi
fi

echo "[INFO] Using GPU ids: $gpu_ids"

python /data/benchmark_metrics/insight/qwen_2511_attention_fullmap_kfull.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref \
  --out_dir /mnt/jfs/qwen-edit-attn-fullmap-keycolor-save-kfull-1-1 \
  --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
  --gpus "$gpu_ids" \
  --key_txt /data/benchmark_metrics/insight/key_folder/analysis_key.txt \
  --steps 28 \
  --true-cfg-scale 4.0 \
  --max-tokens 128 \
  --max-q-tokens 128 \
  --aggregate-head mean \
  --aggregate-block mean \
  --step-stride 1 \
  --block-stride 1 \
  --panel-size 1.1 \
  --image-dpi 240 \
  --ref-labels cref,sref \
  --attn-cmap turbo \
  --attn-gamma 0.48 \
  --high-attn-quantile 0.9 \
  --high-attn-contour-color "#00e5ff" \
  --high-attn-contour-width 1.0 \
  --boundary-linewidth 1.6 \
  --save-attn-tensor \
  --save-format png \
  $extra_args

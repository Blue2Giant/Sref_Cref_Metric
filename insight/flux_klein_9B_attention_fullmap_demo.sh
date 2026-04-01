#!/usr/bin/env bash
set -euo pipefail

overwrite="${OVERWRITE:-0}"
gpu_ids="${1:-${GPU_IDS:-}}"
sref_root="${SREF_ROOT:-/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content}"
key_txt="${KEY_TX:-/data/benchmark_metrics/insight/key_folder/flux/flux_analysis_key.txt}"
out_dir="${OUT_DIR:-$sref_root/flux-klein-9b-attn-fullmap-1-1}"
steps="${STEPS:-4}"
max_tokens="${MAX_TOKENS:-128}"
max_q_tokens="${MAX_Q_TOKENS:-128}"
input_resolution="${INPUT_RESOLUTION:-}"
max_input_long_side="${MAX_INPUT_LONG_SIDE:-1024}"
cache_step_block="${CACHE_STEP_BLOCK:-1}"
cache_full_attn="${CACHE_FULL_ATTN:-1}"
save_attn_tensor="${SAVE_ATTN_TENSOR:-1}"
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
echo "[INFO] Using key file: $key_txt"
echo "[INFO] Output dir: $out_dir"
echo "[INFO] steps=$steps input_resolution=${input_resolution:-<none>} max_input_long_side=$max_input_long_side cache_step_block=$cache_step_block cache_full_attn=$cache_full_attn save_attn_tensor=$save_attn_tensor"

if [ -n "$input_resolution" ]; then
  extra_args="$extra_args --input_resolution $input_resolution"
fi

if [ "${max_input_long_side:-0}" -gt 0 ]; then
  extra_args="$extra_args --max-input-long-side $max_input_long_side"
fi

if [ "$cache_step_block" = "1" ]; then
  extra_args="$extra_args --cache-step-block"
fi

if [ "$cache_full_attn" = "1" ]; then
  extra_args="$extra_args --cache-full-attn"
fi

if [ "$save_attn_tensor" = "1" ]; then
  extra_args="$extra_args --save-attn-tensor"
fi

python /data/benchmark_metrics/sref_cref/flux_klein_9B_attention_fullmap.py \
  --prompts_json "$sref_root/prompts.json" \
  --cref_dir "$sref_root/cref" \
  --sref_dir "$sref_root/sref" \
  --out_dir "$out_dir" \
  --model_name /mnt/jfs/model_zoo/FLUX.2-klein-9B/ \
  --gpus "$gpu_ids" \
  --key_txt "$key_txt" \
  --steps "$steps" \
  --guidance_scale 1.0 \
  --max-tokens "$max_tokens" \
  --max-q-tokens "$max_q_tokens" \
  --aggregate-head mean \
  --aggregate-block mean \
  --step-stride 1 \
  --block-stride 1 \
  --panel-size 1.1 \
  --image-dpi 240 \
  --ref-labels cref,sref \
  --attn-gamma 0.48 \
  --boundary-linewidth 1.2 \
  --save-format png \
  $extra_args

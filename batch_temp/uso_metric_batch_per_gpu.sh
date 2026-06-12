#!/bin/bash
# Usage: uso_metric_batch_per_gpu.sh <GPU_ID> <MODEL_1> [MODEL_2 ...]
# Runs the same encoder/VLM metric pipeline as uso_metric_batch.sh, but pinned
# to a single GPU and a caller-supplied subset of models. Intended to be
# launched in parallel (one process per GPU) so the 4 inference workers do not
# step on each other.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <GPU_ID> <MODEL_1> [MODEL_2 ...]" >&2
  exit 2
fi

GPU="$1"; shift
MODELS=("$@")

cd /data/benchmark_metrics/benchmark_metrics
RUNNER_PY="/data/benchmark_metrics/benchmark_metrics/encoder_batch_runner.py"

SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content"
CONTENT_DIR="$SREF_ROOT/cref"
STYLE_DIR="$SREF_ROOT/sref"
SREF_PROMPT="$SREF_ROOT/prompts.json"

DINOV2_MODEL="/mnt/jfs/model_zoo/dinov2-with-registers-large"
CAS_MODEL="/mnt/jfs/model_zoo/dinov2-base"
ONEIG_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder"
CSD_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder/csd.pth"
CSD_MODEL_ONLY="/mnt/jfs/model_zoo/OneIG-StyleEncoder/vit-b-300ep.pth.tar"
VIT_L="/mnt/jfs/model_zoo/OneIG-StyleEncoder/ViT-L-14.pt"
CLIPCAP_MODEL="/mnt/jfs/model_zoo/clip-vit-large-patch14"

overwrite=1
num_procs=64  # lowered from 128 because up to 4 of these scripts run in parallel

xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
xingpeng_model=qwen3vlw8a8

export CUDA_VISIBLE_DEVICES="$GPU"
# After CUDA_VISIBLE_DEVICES filters to a single device, that device is index 0
# from PyTorch's perspective, so pass --gpus "0" downstream.
GPUS="0"

for MODEL in "${MODELS[@]}"; do
  RESULT_DIR="$SREF_ROOT/$MODEL"
  if [[ ! -d "$RESULT_DIR" ]]; then
    echo "[gpu=$GPU] skip missing model dir: $RESULT_DIR" >&2
    continue
  fi

  OUT_DINOV2_JSON="$RESULT_DIR/dinov2_out.json"
  OUT_CAS_JSON="$RESULT_DIR/cas_out.json"
  OUT_ONEIG_JSON="$RESULT_DIR/oneig_out.json"
  OUT_CLIPCAP_JSON="$RESULT_DIR/clipcap_out.json"
  OUT_CSD_JSON="$RESULT_DIR/csd_out.json"
  OUT_LAION_JSON="$RESULT_DIR/laion_scores.json"
  OUT_V25_AESTHETIC="$RESULT_DIR/v25_scores.json"

  echo "==== [gpu=$GPU] CSD ($MODEL) ===="
  python3 "$RUNNER_PY" pair \
    --encoder csd \
    --dir_a "$STYLE_DIR" \
    --dir_b "$RESULT_DIR" \
    --out_json "$OUT_CSD_JSON" \
    --model dummy \
    --csd_arch vit_base \
    --csd_model_path "$CSD_MODEL_ONLY" \
    --device cuda \
    --gpus "$GPUS" \
    --overwrite "$overwrite"

  echo "==== [gpu=$GPU] oneig ($MODEL) ===="
  python3 "$RUNNER_PY" pair \
    --encoder oneig \
    --dir_a "$STYLE_DIR" \
    --dir_b "$RESULT_DIR" \
    --model dummy \
    --oneig_model_path "$CSD_MODEL" \
    --oneig_se_model_path "$ONEIG_MODEL" \
    --oneig_clip_model_path "$VIT_L" \
    --out_json "$OUT_ONEIG_JSON" \
    --gpus "$GPUS" \
    --overwrite "$overwrite"

  echo "==== [gpu=$GPU] dinov2 ($MODEL) ===="
  python3 "$RUNNER_PY" pair \
    --encoder dinov2 \
    --dir_a "$CONTENT_DIR" \
    --dir_b "$RESULT_DIR" \
    --model "$DINOV2_MODEL" \
    --out_json "$OUT_DINOV2_JSON" \
    --gpus "$GPUS" \
    --overwrite "$overwrite"

  echo "==== [gpu=$GPU] cas ($MODEL) ===="
  python3 "$RUNNER_PY" pair \
    --encoder cas \
    --dir_a "$CONTENT_DIR" \
    --dir_b "$RESULT_DIR" \
    --model "$CAS_MODEL" \
    --out_json "$OUT_CAS_JSON" \
    --gpus "$GPUS" \
    --overwrite "$overwrite"

  echo "==== [gpu=$GPU] clip_t ($MODEL) ===="
  python3 "$RUNNER_PY" clip_t \
    --image_dir "$RESULT_DIR" \
    --prompt_json "$SREF_PROMPT" \
    --out_json "$OUT_CLIPCAP_JSON" \
    --model /mnt/jfs/model_zoo/openai/clip-vit-base-patch32 \
    --sim_metric cosine \
    --clipcap_text_mode first_sentence \
    --overwrite "$overwrite"

  echo "==== [gpu=$GPU] laion aesthetic ($MODEL) ===="
  python3 "$RUNNER_PY" aesthetic \
    --backend laion \
    --image_dir "$RESULT_DIR" \
    --out_json "$OUT_LAION_JSON" \
    --laion_clip_model ViT-L-14 \
    --laion_clip_ckpt /mnt/jfs/model_zoo/open_clip/open_clip_model_ea4f182e96863ce2a27be5067cdb54d4.safetensors \
    --laion_linear_path ~/.cache/emb_reader/sa_0_4_vit_l_14_linear.pth \
    --device cuda \
    --gpus "$GPUS" \
    --overwrite "$overwrite"

  echo "==== [gpu=$GPU] aesthetic v25 ($MODEL) ===="
  python3 "$RUNNER_PY" aesthetic \
    --backend v25 \
    --image_dir "$RESULT_DIR" \
    --out_json "$OUT_V25_AESTHETIC" \
    --v25_encoder_model_name /mnt/jfs/model_zoo/siglip-so400m-patch14-384/ \
    --dtype bfloat16 \
    --device cuda \
    --gpus "$GPUS" \
    --overwrite "$overwrite"

  echo "==== [gpu=$GPU] vlm style ($MODEL) ===="
  output_json_style_discrete="$RESULT_DIR/qwen_resize_output_style_descrete.json"
  reason_json_style_discrete="$RESULT_DIR/qwen_resize_output_style_reason_descrete.json"
  python3 /data/benchmark_metrics/vlm_similarity/style_similarity_dir.py \
    --style_dir "$STYLE_DIR" \
    --output_dir "$RESULT_DIR" \
    --out_score_json "$output_json_style_discrete" \
    --out_reason_json "$reason_json_style_discrete" \
    --base_url "$xingpeng_ip" \
    --model "$xingpeng_model" \
    --num_procs "$num_procs" \
    --overwrite

  echo "==== [gpu=$GPU] vlm content ($MODEL) ===="
  output_json_content_discrete="$RESULT_DIR/qwen_resize_output_content_descrete.json"
  reason_json_content_discrete="$RESULT_DIR/qwen_resize_output_content_reason_descrete.json"
  python3 /data/benchmark_metrics/vlm_similarity/content_similarity_dir.py \
    --content_dir "$CONTENT_DIR" \
    --output_dir "$RESULT_DIR" \
    --out_json "$output_json_content_discrete" \
    --out_reason_json "$reason_json_content_discrete" \
    --base_url "$xingpeng_ip" \
    --model "$xingpeng_model" \
    --num_procs "$num_procs" \
    --overwrite

  echo "==== [gpu=$GPU] vlm instruction follow ($MODEL) ===="
  OUT_SCORE_JSON="$RESULT_DIR/follow_scores.json"
  OUT_REASON_JSON="$RESULT_DIR/follow_reasons.json"
  python3 /data/benchmark_metrics/vlm_similarity/edit_instruction_follow_dir.py \
    --image_dir "$RESULT_DIR" \
    --prompt_json "$SREF_PROMPT" \
    --out_score_json "$OUT_SCORE_JSON" \
    --out_reason_json "$OUT_REASON_JSON" \
    --base_url "$xingpeng_ip" \
    --model "$xingpeng_model" \
    --instruction_text_mode first_sentence \
    --num_procs "$num_procs" \
    --overwrite

  echo "==== [gpu=$GPU] triplet qwen dual judge ($MODEL) ===="
  output_json_content_reject="$RESULT_DIR/qwen_reject_cref.json"
  output_json_style_reject="$RESULT_DIR/qwen_reject_sref.json"
  python3 /data/benchmark_metrics/vlm_similarity/triplet_qwen_dual_judge.py \
    --content_dir "$CONTENT_DIR" \
    --style_dir "$STYLE_DIR" \
    --result_dir "$RESULT_DIR" \
    --output_content_json "$output_json_content_reject" \
    --output_style_json "$output_json_style_reject" \
    --endpoint "${xingpeng_model}@${xingpeng_ip}" \
    --procs_per_endpoint "$num_procs" \
    --overwrite

  echo "==== [gpu=$GPU] DONE model=$MODEL ===="
done

echo "==== [gpu=$GPU] ALL MODELS COMPLETE ===="

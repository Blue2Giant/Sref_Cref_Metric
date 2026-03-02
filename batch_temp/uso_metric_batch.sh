#!/bin/bash
set -euo pipefail
cd /data/benchmark_metrics/benchmark_metrics
RUNNER_PY="/data/benchmark_metrics/benchmark_metrics/encoder_batch_runner.py"
GPUS="0"
MODEL="uso"
SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content"
CONTENT_DIR="$SREF_ROOT/cref"
STYLE_DIR="$SREF_ROOT/sref"
RESULT_DIR="$SREF_ROOT/$MODEL"
SREF_PROMPT="$SREF_ROOT/prompts.json"
#模型权重位置配置
DINOV2_MODEL="/mnt/jfs/model_zoo/dinov2-with-registers-large"
CAS_MODEL="/mnt/jfs/model_zoo/dinov2-base"
ONEIG_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder"
CSD_MODEL="/data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar"
CLIPCAP_MODEL="/mnt/jfs/model_zoo/clip-vit-large-patch14"
ONEALIGN_MODEL="/mnt/jfs/model_zoo/one-align"
ONEALIGN_TASK="aesthetics"

#输出的标注路径配置
OUT_DINOV2_JSON="$RESULT_DIR/dinov2_out.json"
OUT_CAS_JSON="$RESULT_DIR/cas_out.json"
OUT_ONEIG_JSON="$RESULT_DIR/oneig_out.json"
OUT_CLIPCAP_JSON="$RESULT_DIR/clipcap_out.json"
OUT_ONEALIGN_JSON="$RESULT_DIR/onealign_out.json"
OUT_CSD_JSON="$RESULT_DIR/csd_out.json"
OUT_LAION_JSON="$RESULT_DIR/laion_scores.json"
OUT_V25_AESTHETIC="$RESULT_DIR/v25_scores.json"
overwrite=0
# # 风格一致性
# echo "==== CSD ===="
# python3 "$RUNNER_PY" pair \
#   --encoder csd \
#   --dir_a "$STYLE_DIR" \
#   --dir_b "$RESULT_DIR" \
#   --out_json "$OUT_CSD_JSON" \
#   --model dummy \
#   --csd_arch vit_base \
#   --csd_model_path $CSD_MODEL \
#   --device cuda \
#   --gpus "$GPUS" \
#   --overwrite $overwrite \
#     --sim_metric l2

# echo "=== oneig ===="
# python3 "$RUNNER_PY" pair \
#   --encoder oneig \
#   --dir_a "$STYLE_DIR" \
#   --dir_b "$RESULT_DIR" \
#   --model "$ONEIG_MODEL" \
#   --out_json "$OUT_ONEIG_JSON" \
#   --gpus "$GPUS" \
#   --overwrite $overwrite \
#     --sim_metric l2

# #内容一致性
# echo "=== dinov2 ===="
# python3 "$RUNNER_PY" pair \
#   --encoder dinov2 \
#   --dir_a "$CONTENT_DIR" \
#   --dir_b "$RESULT_DIR" \
#   --model "$DINOV2_MODEL" \
#   --out_json "$OUT_DINOV2_JSON" \
#   --gpus "$GPUS" \
#   --overwrite $overwrite
# echo "=== cas ===="
# python3 "$RUNNER_PY" pair \
#   --encoder cas \
#   --dir_a "$CONTENT_DIR" \
#   --dir_b "$RESULT_DIR" \
#   --model "$CAS_MODEL" \
#   --out_json "$OUT_CAS_JSON" \
#   --gpus "$GPUS" \
#   --overwrite $overwrite

# #指令遵循
# echo "=== clip cap ==="
# python3 "$RUNNER_PY" clip_cap \
#   --image_dir "$RESULT_DIR" \
#   --prompt_json "$SREF_PROMPT" \
#   --out_json "$OUT_CLIPCAP_JSON" \
#   --model "$CLIPCAP_MODEL" \
#   --gpus "$GPUS" \
#   --clipcap_text_mode first_sentence \
#   --overwrite $overwrite

# #美学评分
# echo "=== laion aesthetic ==="
# python /data/benchmark_metrics/benchmark_metrics/encoder_batch_runner.py aesthetic \
#   --backend laion \
#   --image_dir $RESULT_DIR \
#   --out_json $OUT_LAION_JSON \
#   --laion_clip_model ViT-L-14 \
#   --laion_clip_ckpt /mnt/jfs/model_zoo/open_clip/open_clip_model_ea4f182e96863ce2a27be5067cdb54d4.safetensors \
#   --laion_linear_path ~/.cache/emb_reader/sa_0_4_vit_l_14_linear.pth \
#   --device cuda \
#   --gpus 0 \
#   --overwrite $overwrite

# echo "==== aesthetic v25 ===="
# python /data/benchmark_metrics/benchmark_metrics/encoder_batch_runner.py aesthetic \
#   --backend v25 \
#   --image_dir $RESULT_DIR \
#   --out_json $OUT_V25_AESTHETIC \
#   --v25_encoder_model_name /mnt/jfs/model_zoo/siglip-so400m-patch14-384/ \
#   --dtype bfloat16 \
#   --device cuda \
#   --gpus 0 \
#   --overwrite $overwrite

# echo "====vlm style===="
# style_dir="$SREF_ROOT/sref"
# result_dir="$SREF_ROOT/$MODEL"
# output_json_style_discrete="$SREF_ROOT/$MODEL/qwen_resize_output_style_descrete.json"
# reason_json_style_discrete="$SREF_ROOT/$MODEL/qwen_resize_output_style_reason_descrete.json"
# xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
# xingpeng_model=qwen3vlw8a8
# python3 /data/benchmark_metrics/vlm_similarity/style_similarity_dir.py \
#   --style_dir $style_dir \
#   --output_dir $result_dir \
#   --out_score_json $output_json_style_discrete \
#   --out_reason_json $reason_json_style_discrete \
#   --base_url $xingpeng_ip \
#   --model $xingpeng_model \
#   --num_procs 32 \
#   --overwrite

# echo "====vlm content===="
# content_dir="$SREF_ROOT/cref"
# result_dir="$SREF_ROOT/$MODEL"
# output_json_content_discrete="$SREF_ROOT/$MODEL/qwen_resize_output_content_descrete.json"
# reason_json_content_discrete="$SREF_ROOT/$MODEL/qwen_resize_output_content_reason_descrete.json"
# xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
# xingpeng_model=qwen3vlw8a8
# python3 /data/benchmark_metrics/vlm_similarity/content_similarity_dir.py \
#   --content_dir $content_dir \
#   --output_dir $result_dir \
#   --out_json $output_json_content_discrete \
#   --out_reason_json $reason_json_content_discrete \
#   --base_url $xingpeng_ip \
#   --model $xingpeng_model \
#   --num_procs 64 \
#   --overwrite

echo "=== vlm insturction follow ==="
OUT_SCORE_JSON="$SREF_ROOT/$MODEL/follow_scores.json"
OUT_REASON_JSON="$SREF_ROOT/$MODEL/follow_reasons.json"
xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
xingpeng_model=qwen3vlw8a8
python3 /data/benchmark_metrics/vlm_similarity/edit_instruction_follow_dir.py \
  --image_dir $SREF_ROOT/$MODEL \
  --prompt_json $SREF_PROMPT \
  --out_score_json $OUT_SCORE_JSON \
  --out_reason_json $OUT_REASON_JSON \
  --base_url $xingpeng_ip \
  --model $xingpeng_model \
  --instruction_text_mode first_sentence \
  --num_procs 32 \
  --overwrite

echo "=== triplet qwen dual judge ==="
content_dir="$SREF_ROOT/cref"
style_dir="$SREF_ROOT/sref"
result_dir="$SREF_ROOT/$MODEL"
output_json_content="$SREF_ROOT/$MODEL/qwen_reject_cref.json"
output_json_style="$SREF_ROOT/$MODEL/qwen_reject_sref.json" 
python3 /data/benchmark_metrics/vlm_similarity/triplet_qwen_dual_judge.py \
    --content_dir "$content_dir" \
    --style_dir "$style_dir" \
    --result_dir "$result_dir" \
    --output_content_json $output_json_content \
    --output_style_json $output_json_style \
    --endpoint "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1" \
    --procs_per_endpoint 32
    
CONTENT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref"
STYLE_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref"
RESULT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
SREF_PROMPT="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json"
SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture"
OUT_SCORE_JSON="$SREF_ROOT/follow_scores.json"
OUT_REASON_JSON="$SREF_ROOT/follow_reasons.json"
xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
xingpeng_model=qwen3vlw8a8
python3 /data/benchmark_metrics/vlm_similarity/edit_instruction_follow_dir.py \
  --image_dir $RESULT_DIR \
  --prompt_json $SREF_PROMPT \
  --out_score_json $OUT_SCORE_JSON \
  --out_reason_json $OUT_REASON_JSON \
  --base_url $xingpeng_ip \
  --model $xingpeng_model \
  --api_key YOUR_KEY \
  --instruction_text_mode first_sentence \
  --num_procs 32 \
  --overwrite
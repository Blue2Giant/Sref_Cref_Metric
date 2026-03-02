CONTENT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref"
STYLE_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref"
RESULT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
OUT_CONTENT_JSON="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/triplet_content_scores.json"
OUT_STYLE_JSON="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/triplet_style_scores.json"
BASE_URL="https://models-proxy.stepfun-inc.com/v1"
MODEL="gpt-4o"
export OPENAI_API_KEY="ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig"
python3 /data/benchmark_metrics/vlm_similarity/triplet_gpt4o_dual_judge.py \
  --content_dir $CONTENT_DIR \
  --style_dir $STYLE_DIR \
  --result_dir $RESULT_DIR \
  --output_content_json $OUT_CONTENT_JSON \
  --output_style_json $OUT_STYLE_JSON \
  --base_url $BASE_URL \
  --model $MODEL \
  --num_procs 8

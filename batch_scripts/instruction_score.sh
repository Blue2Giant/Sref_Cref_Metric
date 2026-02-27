python3 /data/benchmark_metrics/vlm_similarity/edit_instruction_follow_dir.py \
  --image_dir /path/to/images \
  --prompt_json /path/to/prompts.json \
  --out_score_json /path/to/follow_scores.json \
  --out_reason_json /path/to/follow_reasons.json \
  --base_url http://host:port/v1 \
  --api_key YOUR_KEY \
  --model Qwen3-VL-30B-A3B-Instruct \
  --num_procs 8
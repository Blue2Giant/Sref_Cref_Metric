export GEMINI_API_KEY=ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
python /data/benchmark_metrics/caption_pipe/gemini_image_min_batch.py \
  --prompts_json $sref_dir/prompts.json \
  --cref_dir $sref_dir/cref \
  --sref_dir $sref_dir/sref \
  --out_dir $sref_dir/gemini-edit \
  --model_id gemini-2.5-flash-image-native \
  --num_procs 8 \
  --num_generate 3 \
  --overwrite
cref_sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
python /data/benchmark_metrics/caption_pipe/gemini_image_min_batch.py \
  --prompts_json $cref_sref_dir/prompts.json \
  --cref_dir $cref_sref_dir/cref \
  --sref_dir $cref_sref_dir/sref \
  --out_dir $cref_sref_dir/gemini-edit \
  --model_id gemini-2.5-flash-image-native \
  --num_generate 3 \
  --aspect_ratio "1:1" \
  --overwrite \
  --num_procs 8
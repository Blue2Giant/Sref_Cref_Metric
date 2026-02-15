cd /data/LoraPipeline
python /data/LoraPipeline/content_similarity_triplet.py \
  --root s3://lanjinghong-data/loras_eval_qwen_two_lora_with_trigger_new_with_txt \
  --content-root s3://lanjinghong-data/loras_eval_qwen_filtered/ \
  --id-list  /data/LoraPipeline/similarity_stats/qwen_content_ids.txt \
  --gallery-subdir content_100 \
  --backend clip \
  --gpu-ids 4,5,6,7
#  --output-name s3://lanjinghong-data/loras_eval_qwen_two_lora_with_trigger_new_with_txt/loras_triplets_style_similarity.json \

cd /data/LoraPipeline/

#flux similarity
 python /data/LoraPipeline/content_similarity_batches.py \
  --root s3://lanjinghong-data/loras_eval_flux \
  --backend clip \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --num-workers 8 \
  --output-name content_mean.json \
  # --overwrite

# python /data/LoraPipeline/content_similarity_batches.py \
#   --root s3://lanjinghong-data/loras_eval_qwen \
#   --backend clip \
#   --num-workers 8 \
#   --gpu-ids 0,1,2,3,4,5,6,7 \
  # --overwrite
python /data/LoraPipeline/content_similarity_batches.py \
  --root s3://lanjinghong-data/loras_eval_qwen \
  --backend clip \
  --probe-mode content \
  --content-dir-name content_100 \
  --output-name content_mean.json \
  --num-workers 8 \
  --content_similarity.json
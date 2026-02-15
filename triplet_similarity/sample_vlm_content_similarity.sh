python /data/LoraPipeline/triplet_similarity/sample_vlm_content_similarity.py \
  --root s3://lanjinghong-data/loras_eval_flux/ \
  --out  s3://lanjinghong-data/loras_eval_flux_samples_0.1 \
  --bin-width 0.1 \
  --n-per-bin 20 \
  --refs-per-sample 4 \
  --seed 123 \
  --force-png
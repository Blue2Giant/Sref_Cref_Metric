while true; do
  python /data/LoraPipeline/sref_pipeline/triplet_qwen_style_judge.py \
    --root s3://lanjinghong-data/loras_triplets/flux_0111_triplets \
    --num_samples 1000 \
    --seed 42 \
    --out_all s3://lanjinghong-data/loras_triplets/flux_0111_triplets_all/judge_all.json \
    --out_pos s3://lanjinghong-data/loras_triplets/flux_0111_triplets_all/judge_pos.json \
    --out_neg s3://lanjinghong-data/loras_triplets/flux_0111_triplets_all/judge_neg.json
done
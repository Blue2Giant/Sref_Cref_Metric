  python /data/LoraPipeline/utils/get_9_matrix_model_id.py \
    --root s3://lanjinghong-data/loras_eval_qwen \
    --out-dir /data/LoraPipeline/similarity_stats \
    --content-top-pct 50 \
    --style-top-pct 50 \
    --grid-num 9
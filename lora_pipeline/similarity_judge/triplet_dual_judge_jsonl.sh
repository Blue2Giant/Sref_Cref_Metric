input_jsonl=/mnt/jfs/loras_triplets/flux_0215_triplets_latest_9grid_all/triplets_new.jsonl
OUT_DIR=/data/LoraPipeline/assets/flux_mask_valid_0215_9grid_all_strict_jsonl
python3 /data/LoraPipeline/sref_pipeline/triplet_qwen_dual_judge.py \
    --root "$ROOT" \
    --input_jsonl $input_jsonl \
    --out_all "${OUT_DIR}/all.json" \
    --out_pos "${OUT_DIR}/pos.json" \
    --out_neg "${OUT_DIR}/neg.json" \
    --num_samples 0 \
    --out_detail "${OUT_DIR}/detail.json" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.18.35:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.19.47:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.16.19:22002/v1" \
    --content_conf_thr 0.5 \
    --style_conf_thr 0.5 \
    --content_ratio 0.6 \
    --style_ratio 0.6 \
    --procs_per_endpoint 128

    # --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.4.65:22002/v1" \
    # --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.6.95:22002/v1" \
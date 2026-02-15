#!/bin/bash

# 注意：请确认 ROOT 目录下包含 style_and_content/, content_*/, style_*/ 子目录
# 如果目录结构不同，请修改 ROOT 指向正确的路径

mkdir -p "$OUT_DIR"
while true;do

    # ROOT=/mnt/jfs/loras_triplets/illustrious_0203_triplets
    # OUT_DIR=/mnt/jfs/loras_triplets/illustrious_0203_triplets_dual_judge
    # python3 /data/LoraPipeline/sref_pipeline/triplet_qwen_dual_judge.py \
    #     --root "$ROOT" \
    #     --num_samples 1000 \
    #     --out_all "${OUT_DIR}/all.json" \
    #     --out_pos "${OUT_DIR}/pos.json" \
    #     --out_neg "${OUT_DIR}/neg.json" \
    #     --out_detail "${OUT_DIR}/detail.json" \
    #     --endpoint "v1p3@http://stepcast-router.shai-core:9200/v1" \
    #     --endpoint "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1" \
    #     --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.4.65:22002/v1" \
    #     --procs_per_endpoint 32
    ROOT="s3://lanjinghong-data/loras_triplets/flux_0129_triplets"
    OUT_DIR="/data/LoraPipeline/assets/flux_mask"
    ROOT="/mnt/jfs/loras_triplets/flux_0212_triplets"
    OUT_DIR="/data/LoraPipeline/assets/flux_mask_0212_only_human/"
    ROOT="/mnt/jfs/loras_triplets/flux_0214_triplets"
    ROOT="/mnt/jfs/loras_triplets/flux_0214_triplets_latest"
    OUT_DIR="/data/LoraPipeline/assets/flux_mask_latest_0214"
    OUT_DIR="/data/LoraPipeline/assets/flux_mask_latest_0214_ratio0.5"
    OUT_DIR="/data/LoraPipeline/assets/flux_mask_latest_0214_all_tred0.4"
    OUT_DIR="/data/LoraPipeline/assets/flux_mask_latest_0214_all_tred0.45_content_0.3_style_0.3"
    python3 /data/LoraPipeline/sref_pipeline/triplet_qwen_dual_judge.py \
        --root "$ROOT" \
        --num_samples 2000 \
        --out_all "${OUT_DIR}/all.json" \
        --out_pos "${OUT_DIR}/pos.json" \
        --out_neg "${OUT_DIR}/neg.json" \
        --out_detail "${OUT_DIR}/detail.json" \
        --endpoint "v1p3@http://stepcast-router.shai-core:9200/v1" \
        --endpoint "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1" \
        --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.4.65:22002/v1" \
        --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.6.95:22002/v1" \
        --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.191.9.68:22002/v1" \
        --conf_thr 0.45 \
        --content_ratio 0.30 \
        --style_ratio 0.30 \
        --procs_per_endpoint 32
    #        --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.4.65:22002/v1" \
    #     --content_id_txt /data/LoraPipeline/assets/flux_human_content_final.txt \
    # --style_id_txt /data/LoraPipeline/assets/flux_style_1.txt \
    #sdxl
    # ROOT="s3://lanjinghong-data/loras_triplets/sdxl_0203_triplets"
    # OUT_DIR="s3://lanjinghong-data/sdxl_0111_triplets_dual_judge"
    # python3 /data/LoraPipeline/sref_pipeline/triplet_qwen_dual_judge.py \
    #     --root "$ROOT" \
    #     --num_samples 0 \
    #     --out_all "${OUT_DIR}/all.json" \
    #     --out_pos "${OUT_DIR}/pos.json" \
    #     --out_neg "${OUT_DIR}/neg.json" \
    #     --out_detail "${OUT_DIR}/detail.json" \
    #     --num_samples 10 \
    #     --endpoint "v1p3@http://stepcast-router.shai-core:9200/v1" \
    #     --endpoint "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1" \
    #     --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.204.6.95:22002/v1" \
    #     --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.19.61:22002/v1" \
    #     --procs_per_endpoint 16 \
    #     --overwrite
done
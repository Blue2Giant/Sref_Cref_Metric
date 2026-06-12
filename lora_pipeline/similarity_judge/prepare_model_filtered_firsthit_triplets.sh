#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/benchmark_metrics
python_bin=${PYTHON_BIN:-python3}
prepared_dir=${PREPARED_TRIPLET_DIR:-${repo_root}/logs/triplet_jsonl/model_filtered_firsthit}
mkdir -p "${prepared_dir}"

prepare_py="${repo_root}/lora_pipeline/tools/prepare_single_model_triplet_jsonl_for_firsthit.py"
model_id_dir="${repo_root}/lora_pipeline/meta/model_ids/models"

prepare_one() {
    local source_jsonl="$1"
    local model_id_txt="$2"
    local out_jsonl="$3"
    local missing_txt="$4"
    echo "[prepare] source_jsonl=${source_jsonl}"
    echo "[prepare] model_id_txt=${model_id_txt}"
    "${python_bin}" "${prepare_py}" \
        --source-jsonl "${source_jsonl}" \
        --model-id-txt "${model_id_txt}" \
        --out-jsonl "${out_jsonl}" \
        --missing-model-id-txt "${missing_txt}" \
        --progress-every 200
}

prepare_one \
    "${repo_root}/logs/triplet_jsonl/illustrious_0321_two_lora_and_illustrious_s3_eval_images_by_model.jsonl" \
    "${model_id_dir}/illustrious_content.txt" \
    "${prepared_dir}/illustrious_content_model_filtered_triplet.jsonl" \
    "${prepared_dir}/illustrious_content_model_filtered_missing_model_ids.txt"

prepare_one \
    "${repo_root}/logs/triplet_jsonl/illustrious_0321_two_lora_and_illustrious_s3_eval_images_by_model.jsonl" \
    "${model_id_dir}/illustrious_style.txt" \
    "${prepared_dir}/illustrious_style_model_filtered_triplet.jsonl" \
    "${prepared_dir}/illustrious_style_model_filtered_missing_model_ids.txt"

prepare_one \
    "${repo_root}/logs/triplet_jsonl/flux_0321_0326_and_s3_eval_images_by_model.jsonl" \
    "${model_id_dir}/flux_content.txt" \
    "${prepared_dir}/flux_content_model_filtered_triplet.jsonl" \
    "${prepared_dir}/flux_content_model_filtered_missing_model_ids.txt"

prepare_one \
    "${repo_root}/logs/triplet_jsonl/flux_0321_0326_and_s3_eval_images_by_model.jsonl" \
    "${model_id_dir}/flux_style.txt" \
    "${prepared_dir}/flux_style_model_filtered_triplet.jsonl" \
    "${prepared_dir}/flux_style_model_filtered_missing_model_ids.txt"

prepare_one \
    "${repo_root}/logs/triplet_jsonl/qwen_0323_one_lora_and_qwen_s3_eval_images_by_model.jsonl" \
    "${model_id_dir}/qwen_content.txt" \
    "${prepared_dir}/qwen_content_model_filtered_triplet.jsonl" \
    "${prepared_dir}/qwen_content_model_filtered_missing_model_ids.txt"

prepare_one \
    "${repo_root}/logs/triplet_jsonl/qwen_0323_one_lora_and_qwen_s3_eval_images_by_model.jsonl" \
    "${model_id_dir}/qwen_style.txt" \
    "${prepared_dir}/qwen_style_model_filtered_triplet.jsonl" \
    "${prepared_dir}/qwen_style_model_filtered_missing_model_ids.txt"

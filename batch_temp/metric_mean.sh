data_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
MODELS=("newnew800_csgo" "newnew800_easyref" "newnew800_flux_9b" "newnew800_omnistyle" "qwen-edit" "TeleStyle" "uso" "ours")

data_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
MODELS=("uso" "ours" "gpt4o-edit" "gemini-edit" "qwen-edit" "flux_9b_klein")

data_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
MODELS=("newnew800_csgo" "newnew800_easyref" "newnew800_flux_9b" "newnew800_omnistyle" "qwen-edit" "TeleStyle" "uso" "ours" "gpt4o-edit" "gemini-edit" )

for model in "${MODELS[@]}"; do
    jsons=()
    missing=()
    for p in \
        $data_root/$model/dinov2_out.json \
        $data_root/$model/cas_out.json \
        $data_root/$model/oneig_out.json \
        $data_root/$model/clipcap_out.json \
        $data_root/$model/csd_out.json \
        $data_root/$model/laion_scores.json \
        $data_root/$model/v25_scores.json \
        $data_root/$model/qwen_resize_output_style_descrete.json \
        $data_root/$model/qwen_resize_output_content_descrete.json \
        $data_root/$model/follow_scores.json \
        $data_root/$model/qwen_reject_cref.json \
        $data_root/$model/qwen_reject_sref.json; do
        if [ -f "$p" ]; then
            jsons+=("$p")
        else
            missing+=("$p")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        printf "missing jsons (%s):\n" "$model"
        printf "%s\n" "${missing[@]}"
    fi
    if [ ${#jsons[@]} -eq 0 ]; then
        printf "no existing jsons, skip: %s\n" "$model"
        continue
    fi
    python /data/benchmark_metrics/batch_temp/json_means_to_csv.py \
        --jsons "${jsons[@]}" \
        --out_csv $data_root/$model/metrics_mean.csv
    echo "$data_root/$model/metrics_mean.csv"
done

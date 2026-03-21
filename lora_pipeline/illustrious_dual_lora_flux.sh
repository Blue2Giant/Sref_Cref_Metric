# flux dual lora
ip27=10.201.17.43
ip28=10.201.16.61

output_meta_root=s3://lanjinghong-data/loras_eval_flux_debug_1226
lora_root=s3://collect-data-datasets/202510/civitai_file/'Flux.1 D'
output_root=/mnt/jfs/loras_combine/flux_0321_dual_lora
pair_model_id_txt=/data/benchmark_metrics/lora_pipeline/meta/model_ids/flux_style_and_content.txt
prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/OTHER_UNIVERSE_TRIGGER.txt
num_prompts=10
negative_prompt=""

while true; do
    python /data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse_dual_flux.py \
        --lora-root "$lora_root" \
        --meta-root "$output_meta_root" \
        --output-root "$output_root" \
        --pair-model-id-txt "$pair_model_id_txt" \
        --base-model flux1-dev.safetensors \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/flux_dual_lora.json \
        --prompt-txt "$prompt_txt" \
        --comfy-host http://$ip27,http://$ip28 \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "solo" \
        --negative-prompt "$negative_prompt"
done

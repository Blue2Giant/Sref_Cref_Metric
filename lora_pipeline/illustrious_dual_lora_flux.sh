# flux dual lora
ip1=10.201.16.61
ip2=10.201.16.63
ip3=10.201.19.23
ip4=10.201.19.23
ip5=10.201.17.65
ip6=10.201.19.53
ip7=10.201.18.8
ip8=10.201.19.16
ip9=10.201.19.28
ip10=10.201.17.59
ip11=10.201.16.41
ip12=10.201.16.34
ip13=10.201.18.6
ip14=10.201.17.36
ip15=10.201.16.54
ip16=10.201.18.49
ip17=10.201.17.54
ip18=10.201.16.5
ip19=10.201.18.41
ip20=10.201.18.28
ip21=10.201.17.58
ip22=10.201.19.49 
ip23=10.201.19.61
ip24=10.201.16.50
ip25=10.201.17.33
ip26=10.201.16.49
ip27=10.201.17.29
ip28=10.201.19.33

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
        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27,http://$ip28 \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "solo" \
        --negative-prompt "$negative_prompt"
done

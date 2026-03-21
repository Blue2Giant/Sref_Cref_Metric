# flux
ip1=10.201.16.50
ip2=10.201.16.11
ip3=10.201.17.59
ip4=10.201.16.54
ip5=10.201.18.49
ip6=10.201.16.63
ip7=10.201.17.65
ip8=10.201.17.54
ip9=10.201.17.34
ip10=10.201.17.58
ip11=10.201.18.28
ip12=10.201.19.23
ip13=10.201.19.53
ip14=10.201.19.41
ip15=10.201.19.33
ip16=10.201.18.8
ip17=10.201.16.34
ip18=10.201.19.28
ip19=10.201.16.61
ip20=10.201.18.41
ip21=10.201.17.43
ip22=10.201.17.66
ip23=10.201.16.49
ip24=10.201.17.33
ip25=10.201.19.16
ip26=10.201.16.5
ip27=10.201.17.36

output_civitai_flux=s3://lanjinghong-data/loras_eval_flux_debug_1226
civitai_flux_loras=s3://collect-data-datasets/202510/civitai_file/'Flux.1 D'
output_root=/mnt/jfs/loras_combine/flux_0321_one_lora
character_prompt_txt=/data/LoraPipeline/scripts/CHARACTER_UNIVERSE_TRIGGER.txt
other_prompt_txt=/data/LoraPipeline/scripts/OTHER_UNIVERSE_TRIGGER.txt
style_prompt_txt=/data/LoraPipeline/scripts/STYLE_UNIVERSE_TRIGGER.txt
character_model_id=/data/benchmark_metrics/lora_pipeline/meta/model_ids/character_flux.txt
others_model_id=/data/benchmark_metrics/lora_pipeline/meta/model_ids/other_flux.txt
style_model_id=/data/benchmark_metrics/lora_pipeline/meta/model_ids/flux_style_1.txt
num_prompts=20
negative_prompt="lowres, normal quality, worst quality, low quality, jpeg artifacts, compression artifacts, pixelated, blurry, out of focus, soft focus, bad contrast, color banding, posterization, chromatic aberration, aliasing, moire, overexposed, underexposed, blown highlights, crushed shadows, noise, watermark, logo, text, caption, signature, username, copyright, bad anatomy, malformed, disfigured, deformed, bad proportions, extra limbs, missing limbs, duplicate body parts, extra digits, missing fingers, fused fingers, webbed fingers, bad hands, bad feet, distorted face, asymmetrical eyes, cross-eye, extra face, cloned person, body cut off, cropped, floating objects, disconnected limbs, perspective errors, depth errors, incorrect shadows, inconsistent lighting, repeated patterns, mirror artifacts"

while true; do
    python /data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse_flux.py \
        --lora-root "$civitai_flux_loras" \
        --meta-root "$output_civitai_flux" \
        --output-root "$output_root" \
        --base-model flux1-dev.safetensors \
        --filter-model-id $character_model_id \
        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27 \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/flux_full_lora-2.json \
        --prompt-txt "$character_prompt_txt" \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --num-prompts $num_prompts \
        --prefix-phrase "solo" \
        --negative-prompt "$negative_prompt" \
        --negative-node-id 43

    python /data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse_flux.py \
        --lora-root "$civitai_flux_loras" \
        --meta-root "$output_civitai_flux" \
        --output-root "$output_root" \
        --base-model flux1-dev.safetensors \
        --filter-model-id $others_model_id \
        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27 \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/flux_full_lora-2.json \
        --prompt-txt "$other_prompt_txt" \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --num-prompts $num_prompts \
        --prefix-phrase "" \
        --negative-prompt "$negative_prompt" \
        --negative-node-id 43

    python /data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse_flux.py \
        --lora-root "$civitai_flux_loras" \
        --meta-root "$output_civitai_flux" \
        --output-root "$output_root" \
        --base-model flux1-dev.safetensors \
        --filter-model-id $style_model_id \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/flux_full_lora-2.json \
        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27 \
        --prompt-txt "$style_prompt_txt" \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --num-prompts $num_prompts \
        --prefix-phrase "" \
        --negative-prompt "$negative_prompt" \
        --negative-node-id 43 \
        --overwrite
done

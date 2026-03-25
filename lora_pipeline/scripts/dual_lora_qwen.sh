ip1=10.191.13.9
ip2=10.191.2.9
output_meta_root=s3://lanjinghong-data/loras_eval_qwen
lora_root=/mnt/jfs/all_loras/civitai/Qwen
output_root=/mnt/jfs/loras_combine/qwen_0323_dual_lora
prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/TRIPLET_UNIVERSE_TRIGGER_NO_UNDERLINE.txt
while true;do
    python /data/benchmark_metrics/lora_pipeline/dual_lora_qwen.py \
    --lora-root $lora_root \
    --meta-root $output_meta_root \
    --output-root $output_root \
    --pair-model-id-txt /data/benchmark_metrics/lora_pipeline/meta/model_ids/qwen_style_and_content.txt \
    --base-model qwen_image_fp8_e4m3fn.safetensors \
    --prompt-txt "" \
    --comfy-host http://$ip1,http://$ip2 \
    --num-workers 4 \
    --num-prompts 200 \
    --download-workers 4 \
    --negative-prompt ""
done
# 每个lora用固定的100个prompt生成内容，用于筛选lora的内容和风格的稳定性
# 查看和生成带有trigger word的
cd /data/LoraPipeline/
ip=10.191.11.23

#flux eval_images的生成
civitai_bucket_flux="s3://lanjinghong-data/civitai_label_binary_classfication_using_prompt_example_filtered_flux/"
civitai_flux_loras=s3://collect-data-datasets/202510/civitai_file/"Flux.1 D"
output_civitai_flux="s3://lanjinghong-data/loras_eval_flux"
ip1=10.201.18.29
ip2=10.201.16.57
ip3=10.201.16.58
ip4=10.201.19.58
ip5=10.201.16.44
ip6=10.201.17.59
ip7=10.191.3.6
ip8=10.191.23.12


python /data/LoraPipeline/check_trigger.py \
  --lora-root s3://collect-data-datasets/202510/civitai_file/Qwen \
  --meta-root s3://lanjinghong-data/civitai_label_binary_classfication_using_prompt_example_filtered_qwen \
  --output-root s3://lanjinghong-data/loras_eval_qwen_check_trigger \
  --prompt-txt /data/LoraPipeline/assets/diverse_prompts_100.txt \
  --num-workers 8 \
#   --overwrite
#如果要覆盖就加上 --overwrite
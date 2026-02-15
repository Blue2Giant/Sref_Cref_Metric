python3 /data/LoraPipeline/utils/collect_checkpoint_version.py \
  --root s3://lanjinghong-data/loras_eval_sdxl \
  --ids-txt /data/LoraPipeline/assets/sdxl_style.txt \
  --output-json /data/LoraPipeline/output/illustrious_style_model_version.json
python3 /data/LoraPipeline/utils/collect_checkpoint_version.py \
  --root s3://lanjinghong-data/loras_eval_sdxl \
  --ids-txt /data/LoraPipeline/assets/sdxl_content.txt \
  --output-json /data/LoraPipeline/output/illustrious_content_model_version.json
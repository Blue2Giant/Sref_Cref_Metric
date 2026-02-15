#flux pair
# /mnt/jfs/loras_triplets/flux_0212_triplets
# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root /mnt/jfs/loras_combine/flux_0111 \
#   --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
#   --out_root     /mnt/jfs/loras_triplets/flux_0213_triplets_positive \
#   --sample_images_per_pair 1 \
#   --output_prompt_json /mnt/jfs/loras_triplets/flux_0212_triplets_positive/prompts.json \
#   --workers 32 \

  # --pair-ids /data/LoraPipeline/assets/flux_mask_0212/pos.json \
# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root /mnt/jfs/loras_combine/flux_0111 \
#   --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
#   --out_root     /mnt/jfs/loras_triplets/flux_0129_triplets \
#   --sample_images_per_pair 1 \
#   --output_prompt_json /mnt/jfs/loras_triplets/flux_0129_triplets/prompts.json \
#   --workers 32 \
#   --pair-ids /data/LoraPipeline/output/pos.txt \

# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root /mnt/jfs/loras_combine/flux_0111 \
#   --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
#   --out_root     /mnt/jfs/loras_triplets/flux_0214_triplets_latest \
#   --sample_images_per_pair 1 \
#   --content_ids_txt /data/LoraPipeline/assets/flux_human_content_final.txt \
#   --style_ids_txt /data/LoraPipeline/assets/flux_style_1.txt \
#   --output_prompt_json /mnt/jfs/loras_triplets/flux_0214_triplets_latest/prompts.json \
#   --workers 32

  # --pair-ids /data/LoraPipeline/output/pos.txt \
# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root s3://lanjinghong-data/loras_combine/flux_0129_triplet_batch \
#   --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
#   --out_root     /mnt/jfs/loras_triplets/flux_0129_triplets \
#   --sample_images_per_pair 20 \
#   --output_prompt_json /mnt/jfs/loras_triplets/flux_0129_triplets/prompts.json \
#   --pair-ids /data/LoraPipeline/output/pos.txt \

#illustrious 
python /data/LoraPipeline/utils/triplet_copy.py \
  --combine_root s3://lanjinghong-data/loras_combine/illustrious_10x10_new_1 \
  --demo_root   s3://lanjinghong-data/loras_eval_illustrious_one_img_magic \
  --out_root     /mnt/jfs/loras_triplets/illustrious_0213_triplets_10x10_new_2 \
  --sample_images_per_pair 1 \
  --output_prompt_json s3://lanjinghong-data/illustrious_0213_triplets/prompts.json \
  --overwrite

# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root s3://lanjinghong-data/loras_combine/flux_0111 \
#   --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
#   --out_root     s3://lanjinghong-data/loras_triplets/flux_0111_triplets \
#   --sample_images_per_pair \
#   --overwrite
#flux
# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root s3://lanjinghong-data/loras_combine/flux_0111 \
#   --demo_root    s3://lanjinghong-data/loras_combine/flux_0111 \
#   --out_root     "s3://lanjinghong-data/loras_combine/sdxl_0111" \
#   --sample_images_per_pair 1 \
#   --workers 32 \
#   --overwrite

#sdxl
# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root s3://lanjinghong-data/loras_combine/sdxl_0203 \
#   --demo_root    s3://lanjinghong-data/loras_eval_sdxl \
#   --out_root     s3://lanjinghong-data/loras_triplets/sdxl_0203_triplets \
#   --sample_images_per_pair 1 \
#   --output_prompt_json s3://lanjinghong-data/loras_triplets/sdxl_0203_triplets/prompts.json \
#   --workers 32 \
#   --overwrite

# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root s3://lanjinghong-data/loras_combine/sdxl_0129 \
#   --demo_root    s3://lanjinghong-data/loras_eval_sdxl \
#   --out_root     s3://lanjinghong-data/loras_triplets/sdxl_0129_triplets \
#   --sample_images_per_pair 1 \
#   --workers 32 \
#   --overwrite
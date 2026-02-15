# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root /mnt/jfs/loras_combine/flux_0111 \
#   --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
#   --out_root     /mnt/jfs/loras_triplets/flux_0214_triplets_humanjudge_all \
#   --sample_images_per_pair 1 \
#   --pair-ids /data/LoraPipeline/assets/flux_mask_latest_0214/all.json \
#   --output_prompt_json /mnt/jfs/loras_triplets/flux_0214_triplets_humanjudge_all/prompts.json \
#   --workers 32
python /data/LoraPipeline/utils/triplet_copy.py \
  --combine_root /mnt/jfs/loras_combine/flux_0111 \
  --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
  --out_root     /mnt/jfs/loras_triplets/flux_mask_latest_0214_ratio0.5 \
  --sample_images_per_pair 1 \
  --pair-ids   /data/LoraPipeline/assets/flux_mask_latest_0214_ratio0.5/pos.json \
  --output_prompt_json /mnt/jfs/loras_triplets/flux_0214_triplets_humanjudge_ratio0.5/prompts.json \
  --workers 32

# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root /mnt/jfs/loras_combine/flux_0111 \
#   --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
#   --out_root     /mnt/jfs/loras_triplets/flux_0214_triplets_humanjudge_neg \
#   --sample_images_per_pair 1 \
#   --pair-ids /data/LoraPipeline/assets/flux_mask_latest_0214/neg.json \
#   --output_prompt_json /mnt/jfs/loras_triplets/flux_0214_triplets_humanjudge_neg/prompts.json \
#   --workers 32

# python /data/LoraPipeline/utils/triplet_copy.py \
#   --combine_root /mnt/jfs/loras_combine/flux_0111 \
#   --demo_root    s3://lanjinghong-data/loras_eval_flux_debug_1226 \
#   --out_root     /mnt/jfs/loras_triplets/flux_0214_triplets_humanjudge \
#   --sample_images_per_pair 1 \
#   --pair-ids /data/LoraPipeline/assets/flux_mask_latest_0214/pos.json \
#   --output_prompt_json /mnt/jfs/loras_triplets/flux_0214_triplets_humanjudge/prompts.json \
#   --workers 32


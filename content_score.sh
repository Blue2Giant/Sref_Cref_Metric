cd /data/LoraPipeline
input_dir="/mnt/jfs/outputs/ill_nyanmix"
probe_image="/data/ComfyKit/illustrious_images/Daphne_Blake_Scooby_Doo_Mystery_Incorporated-000005.png"
python /data/LoraPipeline/content_score.py \
  --gallery_dir $input_dir \
  --probe_image $probe_image \
  --backend dino \
  --output_json gallery_content_scores.json
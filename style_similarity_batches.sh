cd /data/LoraPipeline
export SIGLIP_PATH=/data/USO/weights/siglip
dinov2_reg=/mnt/jfs/model_zoo/dinov2-with-registers-large
export CUDA_VISIBLE_DEVICES=0

input_dir="s3://lanjinghong-data/loras_eval_qwen"
probe_image="/data/ComfyKit/illustrious_images/Daphne_Blake_Scooby_Doo_Mystery_Incorporated-000005.png"
#给flux这边计算风格相似度
python /data/LoraPipeline/style_similarity_batches.py \
  --root s3://lanjinghong-data/loras_eval_flux \
  --pt_style oneig,siglip,csd \
  --model_path /data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar \
  --styleshot_clip_path /mnt/jfs/model_zoo/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --styleshot_weight_path /mnt/jfs/model_zoo/StyleShot/StyleShot/pretrained_weight/style_aware_encoder.bin \
  --oneig_path /mnt/jfs/model_zoo/OneIG-StyleEncoder \
  --encoder_weights 0.35,0.0,0.65 \
  --num-workers 8 \
  # --overwrite
#计算qwen3打的qwen3的prompt的生成结果的相似度
python /data/LoraPipeline/style_similarity_batches.py \
  --root s3://lanjinghong-data/loras_eval_qwen \
  --pt_style oneig,siglip,csd \
  --encoder_weights 0.35,0.0,0.65 \
  --style-mean \
  --style-dir-name style_100 \
  --num-workers 2 \
  --output-name style_mean.json \
  --model_path /data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar \
  --styleshot_clip_path /mnt/jfs/model_zoo/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --styleshot_weight_path /mnt/jfs/model_zoo/StyleShot/StyleShot/pretrained_weight/style_aware_encoder.bin \
  --oneig_path /mnt/jfs/model_zoo/OneIG-StyleEncoder \
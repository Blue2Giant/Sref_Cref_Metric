cd /data/LoraPipeline
export SIGLIP_PATH=/data/USO/weights/siglip
dinov2_reg=/mnt/jfs/model_zoo/dinov2-with-registers-large
export CUDA_VISIBLE_DEVICES=0

input_dir="/mnt/jfs/outputs/ill_nyanmix"
probe_image="/data/ComfyKit/illustrious_images/Daphne_Blake_Scooby_Doo_Mystery_Incorporated-000005.png"
python /data/LoraPipeline/style_score.py \
  --gallery_dir $input_dir \
  --probe_image $probe_image \
  --pt_style oneig,siglip,csd \
  --model_path /data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar \
  --styleshot_clip_path /mnt/jfs/model_zoo/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --styleshot_weight_path /mnt/jfs/model_zoo/StyleShot/StyleShot/pretrained_weight/style_aware_encoder.bin \
  --oneig_path /mnt/jfs/model_zoo/OneIG-StyleEncoder \
  --encoder_weights 0.5,0.4,0.1
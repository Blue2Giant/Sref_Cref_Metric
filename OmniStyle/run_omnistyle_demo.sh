#!/usr/bin/env bash
export T5=/data/USO/weights/t5-xxl
export CLIP=/mnt/jfs/model_zoo/clip-vit-large-patch14/
export FLUX_DEV=/data/USO/weights/FLUX.1-dev/flux1-dev.safetensors
export AE=/data/USO/weights/FLUX.1-dev/ae.safetensors
export LORA=/data/Sref_Cref/OmniStyle/pretrained/dit_lora.safetensors
export SIGLIP_PATH=/data/USO/weights/siglip
cd /data/benchmark_metrics/OmniStyle
sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
python /data/benchmark_metrics/OmniStyle/run_omnistyle.py \
  --prompts_json $sref_root/prompts.json \
  --cref_dir $sref_root/cref \
  --sref_dir $sref_root/sref \
  --out_dir $sref_root/omnistyle \
  --model_type flux-dev \
  --num_steps 25 \
  --guidance 4.0 \
  --gpus 0 \
  --save_jsonl

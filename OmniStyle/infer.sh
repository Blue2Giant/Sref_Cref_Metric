export T5=/data/USO/weights/t5-xxl
export CLIP=/mnt/jfs/model_zoo/clip-vit-large-patch14/
export FLUX_DEV=/data/USO/weights/FLUX.1-dev/flux1-dev.safetensors
export AE=/data/USO/weights/FLUX.1-dev/ae.safetensors
export LORA=/data/Sref_Cref/OmniStyle/pretrained/dit_lora.safetensors
export SIGLIP_PATH=/data/USO/weights/siglip
cd /data/benchmark_metrics/OmniStyle
python infer_demo.py \
       /data/USO/input_demo/image.png \
       /data/USO/input_demo/image_style.png \
       /data/Depth-Anything/external/OmniStyle/result/stylized.jpg
# #!/bin/bash
while true; do
    brainctl launch --charged-group=l40s_yangtong --private-machine=yes \
    --cpu=40 --gpu=8 --memory=360000 --positive-tags "L40S" \
    --image=hub.i.basemind.com/text2image/comfyui-server:0.1 \
    --mount=juicefs+s3://oss.i.shaipower.com/lanjinghong-data:/mnt/jfs \
    --volume /data:/data \
    --i-know-i-am-wasting-resource=false \
    --host-network=true \
    --custom-resources rdma/mlnx_shared=8 --entrypoint="" -- bash /data/benchmark_metrics/lora_pipeline/meta/comfyui_start_new_server.sh
    echo "rlaunch 任务已退出，3秒后重试..."
    sleep 3
done
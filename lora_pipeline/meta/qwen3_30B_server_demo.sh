
while true;do
    brainctl launch --charged-group="maintain" --private-machine=yes \
        --cpu="128" --gpu="4" --memory="1000000" \
        --charged-group="buffer" --positive-tags "H100,H200,H800,L40S" \
        --negative-tags gpu-h100-0107.host.platform.shaipower.com \
        --positive-tags gpu-h800-0187.host.platform.shaipower.com \
        --image=hub.i.basemind.com/stepcast/stepcast:openvllm-qwen3vl-0925 \
        --mount=juicefs+s3://oss.i.shaipower.com/lanjinghong-data:/mnt/jfs \
        --volume /data/:/data/ \
        --host-network=true \
        --enable-sshd=false \
        --entrypoint="" -- bash /data/benchmark_metrics/lora_pipeline/meta/start_qwen_server.sh
done
#--charged-group="buffer" --positive-tags "H100,H200,L40S,H800" \
#        --charged-group="l40s_yangtong"  --positive-tags "L40S" \

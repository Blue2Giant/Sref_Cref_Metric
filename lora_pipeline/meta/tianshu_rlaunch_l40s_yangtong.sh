GROUP="l40s_yangtong"
POSITIVE_TAGS="BI-V150"
CPU="80"
GPU="8"
MEMORY="200000"

trap "exit 0" INT

while getopts "c:g:m:G:P:" opt; do
  case "$opt" in
    c) CPU="$OPTARG" ;;
    g) GPU="$OPTARG" ;;
    m) MEMORY="$OPTARG" ;;
    G)
      case "$OPTARG" in
        1) GROUP="l40s_yangtong" ;;
        2) GROUP="tianshu_test" ;;
      esac
      ;;
    P)
      case "$OPTARG" in
        1) POSITIVE_TAGS="BI-V150" ;;
        2) POSITIVE_TAGS="L40S" ;;
        *) POSITIVE_TAGS="$OPTARG" ;;
      esac
      ;;
  esac
done

shift $((OPTIND -1))

while true; do
  brainctl launch --charged-group=$GROUP --private-machine=yes \
    --cpu="$CPU" --gpu="$GPU" --memory="$MEMORY" \
    --image=hub.i.basemind.com/text2image/tianshu-comfyui-dev:v0.1 \
    --positive-tags "$POSITIVE_TAGS" \
    --max-wait-duration=120m0s   --image-check-timeout=1m0s     --mount=juicefs+s3://oss.i.shaipower.com/lanjinghong-data:/mnt/jfs \
    --volume /data:/data     --i-know-i-am-wasting-resource=false       --enable-sshd=false \
    --entrypoint= -- bash /data/ComfyKit/tianshu_launch/tianshu_server.sh
done

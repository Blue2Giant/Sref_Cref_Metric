#!/usr/bin/env bash
set +e
echo "__STATUS_START__ $(date -Is) host=$(hostname)"
echo "__PS__"
ps -ef | grep -E 'run_uso_gpu_temp2_single|batch_run[.]sh|batch_simple_demo[.]py|supervise_5030|multi_cref_eval' | grep -v grep || true
echo "__GPU_APPS__"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo "__GPU__"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo "__LOGS__"
latest=$(ls -td /data/benchmark_metrics/logs/uso_gpu_temp2_single_* 2>/dev/null | head -1)
echo "latest=$latest"
if [ -n "$latest" ] && [ -f "$latest/gpu0_shard0.log" ]; then
  stat -c 'log_size=%s log_mtime=%y' "$latest/gpu0_shard0.log"
  tail -40 "$latest/gpu0_shard0.log"
fi
echo "__PROC_IO__"
pid=$(pgrep -f 'batch_simple_demo[.]py' | head -1)
echo "pid=$pid"
if [ -n "$pid" ]; then
  ps -o pid,ppid,lstart,etime,stat,pcpu,pmem,rss,vsz,args -p "$pid" || true
  cat "/proc/$pid/io" 2>/dev/null || true
  echo "fd_target_sample:"
  ls -l "/proc/$pid/fd" 2>/dev/null | grep -E 'pytorch_model|safetensors|model' | head -20 || true
fi
echo "__OUTPUT_DIMS__"
python3 - <<'PY'
from pathlib import Path
from PIL import Image
from collections import Counter
import time
root=Path('/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/uso')
dims=Counter(); recent=0; recent_dims=Counter(); samples=[]
cut=time.time()-3600
for p in root.glob('*.png'):
    try:
        st=p.stat(); im=Image.open(p); size=im.size
        dims[size]+=1
        if st.st_mtime>=cut:
            recent+=1; recent_dims[size]+=1
            if len(samples)<5: samples.append((p.name,size,time.strftime('%F %T', time.localtime(st.st_mtime))))
    except Exception as e:
        dims[(str(e),)] += 1
print('total', sum(dims.values()), 'dims', dims.most_common(20))
print('recent_1h', recent, 'recent_dims', recent_dims.most_common(20), 'samples', samples)
PY
echo "__STATUS_END__ $(date -Is)"

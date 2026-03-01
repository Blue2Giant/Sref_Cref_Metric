cd /data/benchmark_metrics/vault
# UV_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple 
# eval $(curl -s http://deploy.i.shaipower.com/httpproxy)

#下载vault
# pip install -i http://mirrors.i.basemind.com/pypi/simple/ --trusted-host mirrors.i.basemind.com "$(megfile cat s3+b://ruiwang/pypi/vault/latest.txt | xargs -I {} sh -c 'megfile cp s3+b://ruiwang/pypi/vault/{} /tmp/{} >/dev/null && echo /tmp/{}')"
# uv run python /data/benchmark_metrics/vault/sample_pics.py --index-url https://pypi.tuna.tsinghua.edu.cn/simple 
export LD_PRELOAD=./fcntl_hack.so #只读系统避免出错
python sample_pics.py
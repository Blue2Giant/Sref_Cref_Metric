cd /data/benchmark_metrics/vault
UV_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple 
eval $(curl -s http://deploy.i.shaipower.com/httpproxy)
uv run python /data/benchmark_metrics/vault/sample_pics.py --index-url https://pypi.tuna.tsinghua.edu.cn/simple 
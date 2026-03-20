ip1=10.201.19.23
ip2=10.201.16.5
ip3=10.201.17.34
ip4=10.201.17.59
ip5=10.201.17.65
ip6=10.201.17.58
ip7=10.201.16.63
ip8=10.201.18.49
ip9=10.201.16.54
ip10=10.201.17.54
ip11=10.201.16.50
ip12=10.201.17.36
ip13=10.201.19.33
ip14=10.201.19.49
ip15=10.201.17.66
ip16=10.201.16.41
ip17=10.201.17.33
ip18=10.201.19.41
ip19=10.201.19.16
ip20=10.201.18.8
ip21=10.201.16.49
ip22=10.201.19.28
ip23=10.201.18.41
ip24=10.201.16.61
ip25=10.201.17.43
ip26=10.201.16.34
cd /data/benchmark_metrics/lora_pipeline
python /data/benchmark_metrics/lora_pipeline/comfykit_demo_stress.py \
  --hosts $ip1,$ip2,$ip3,$ip4,$ip5,$ip6,$ip7,$ip8,$ip9,$ip10,$ip11,$ip12,$ip13,$ip14,$ip15,$ip16,$ip17,$ip18,$ip19,$ip20,$ip21,$ip22,$ip23,$ip24,$ip25,$ip26 \
  --start-port 8188 \
  --port-count 8 \
  --requests-per-port 10 \
  --concurrency 0 \
  --session-pool-size 1 \
  --output-file /data/benchmark_metrics/logs/comfykit_downloads/comfykit_stress_results.json \
  --download-dir /data/benchmark_metrics/logs/comfykit_downloads
import argparse
import asyncio
import json
import random
from dataclasses import asdict

from high_concurrency_workflow import IllustriousSimpleWorkflowProcessor


async def main():
    parser = argparse.ArgumentParser(description="HighConcurrencyWorkflowProcessor 单线程闭环示例")
    parser.add_argument("--endpoints", type=str, default="http://10.201.19.23:8188", help="逗号分隔 endpoint 列表")
    parser.add_argument("--workflow-json", type=str, default="/data/benchmark_metrics/lora_pipeline/meta/illustrious_simple.json")
    parser.add_argument("--save-dir", type=str, default="/data/benchmark_metrics/logs/comfykit_downloads/demo")
    parser.add_argument("--prefix", type=str, default="ComfyUI")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--positive", type=str, default="masterpiece, best quality, 1girl")
    parser.add_argument("--negative", type=str, default="lowres, worst quality")
    parser.add_argument("--loraname", type=str, default="Smooth_Booster_v4.safetensors")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--downloader-pool-size", type=int, default=2)
    parser.add_argument("--download-queue-size", type=int, default=128)
    parser.add_argument("--download-timeout-sec", type=int, default=120)
    parser.add_argument("--wait-timeout-sec", type=int, default=120)
    args = parser.parse_args()

    endpoints = [x.strip() for x in args.endpoints.split(",") if x.strip()]
    seed = args.seed if args.seed > 0 else random.randint(1, (1 << 63) - 1)

    async with IllustriousSimpleWorkflowProcessor(
        endpoints=endpoints,
        comfykit_session_pool_size=1,
        acquire_blocking=True,
        acquire_retries=3,
        acquire_retry_interval_sec=0.5,
        downloader_pool_size=args.downloader_pool_size,
        download_queue_size=args.download_queue_size,
        download_timeout_sec=args.download_timeout_sec,
    ) as processor:
        task_ids = await processor.execute(
            workflow_json_path=args.workflow_json,
            prompt=args.prompt,
            positive=args.positive,
            negative=args.negative,
            random_seed=seed,
            prefix=args.prefix,
            save_dir=args.save_dir,
            optional_fields={"loraname": args.loraname},
        )
        results = await processor.wait_download_tasks(task_ids, timeout_sec=args.wait_timeout_sec)

    print(json.dumps({"task_ids": task_ids}, ensure_ascii=False, indent=2))
    print(json.dumps({k: asdict(v) for k, v in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

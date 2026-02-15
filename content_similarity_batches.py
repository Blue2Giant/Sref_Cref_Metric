#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多进程版本：

遍历 root 下所有子目录（model_dir）：
- 若 model_dir 同时存在 demo_images/ 和 eval_images/：
    * 将 demo_images/ 中的每一张图片都作为 probe_image
    * 对每个 probe_image 调用 content_score.compute_gallery_similarity_megfile(
          eval_images, probe_image, backend, processor=model_processor, model=model
      )
      得到该 probe 相对于 eval_images 的 mean_similarity
    * 对所有 probe 的 mean_similarity 再取平均，写入该子目录下的输出 JSON

新增：
- 可选使用 content_100（或 --content-dir-name 指定的目录）中的每张图片作为 probe：
    * 通过 --probe-mode=content 开启
    * gallery 仍然是 eval_images
    * 其它逻辑与 demo_images 模式一致

并行逻辑：
- 使用 --num-workers 控制进程数
- 每个进程绑定到一个 GPU（通过 --gpu-ids 或自动检测），每个进程各自加载一份模型
- 每个进程处理一部分 model_dir 子目录
"""

import os
import json
import argparse
from typing import List, Dict
import multiprocessing as mp

import torch
from megfile.smart import smart_makedirs, smart_open as mopen

import content_score  # 这里就是你贴的第二个脚本所在模块（含 compute_gallery_similarity_megfile 等）


# ==================== 工具函数 ====================

def join_path(root: str, name: str) -> str:
    return root.rstrip("/") + "/" + name.lstrip("/")


def split_into_chunks(lst: List[str], n: int) -> List[List[str]]:
    """
    把 lst 均匀切成 n 份（最后几份长度差最多 1）
    """
    if n <= 0:
        return [lst]
    total = len(lst)
    if total == 0:
        return []
    n = min(n, total)
    base, extra = divmod(total, n)
    chunks: List[List[str]] = []
    start = 0
    for i in range(n):
        length = base + (1 if i < extra else 0)
        end = start + length
        chunks.append(lst[start:end])
        start = end
    return chunks


def process_single_model_dir(
    model_dir: str,
    backend_name: str,
    processor,
    model,
    output_name: str,
    overwrite: bool,
    probe_mode: str,
    content_dir_name: str,
) -> None:
    """
    在一个进程内部处理单个 model_dir：

    probe_mode = "demo":
      - demo_images 下每一张图作为 probe
      - 分别与 eval_images 比较得到 mean_similarity

    probe_mode = "content":
      - content_dir_name (默认 content_100) 下每一张图作为 probe
      - 分别与 eval_images 比较得到 mean_similarity

    共同：
      - 对所有 probe 的 mean_similarity 求平均，写入 JSON
    """
    demo_dir = join_path(model_dir, "demo_images")
    eval_dir = join_path(model_dir, "eval_images")
    content_dir = join_path(model_dir, content_dir_name)
    out_json = join_path(model_dir, output_name)

    # 选择 probe 的目录
    if probe_mode == "demo":
        probe_dir = demo_dir
    elif probe_mode == "content":
        probe_dir = content_dir
    else:
        print(f"[ERROR] {model_dir}: 未知 probe_mode={probe_mode}，跳过")
        return

    # 检查 eval_images 是否存在（所有模式都需要）
    if not content_score.dir_exists_megfile(eval_dir):
        print(f"[SKIP] {model_dir}: 缺少 eval_images，跳过")
        return

    # 检查 probe 目录是否存在
    if not content_score.dir_exists_megfile(probe_dir):
        print(f"[SKIP] {model_dir}: probe_dir={probe_dir} 不存在，跳过（probe_mode={probe_mode}）")
        return

    # 是否已存在结果
    if (not overwrite) and content_score.smart_exists(out_json):
        print(f"[SKIP] {model_dir}: {out_json} 已存在（未指定 --overwrite），跳过")
        return

    # probe_dir 里所有图片
    probe_imgs = content_score.list_images_recursive_megfile(probe_dir)
    if not probe_imgs:
        print(f"[WARN] {model_dir}: {probe_dir} 下没有图片，跳过（probe_mode={probe_mode}）")
        return

    print(
        f"[INFO] {model_dir}: 在 probe_dir={probe_dir} 下共找到 {len(probe_imgs)} 张图片，"
        f"将全部作为 probe（probe_mode={probe_mode}）参与计算"
    )

    per_probe_mean: Dict[str, float] = {}
    total_mean = 0.0
    valid_probes = 0

    for probe_image in probe_imgs:
        print(f"[INFO] {model_dir}: 以 probe_image={probe_image} 计算 eval_images 相似度")
        try:
            result = content_score.compute_gallery_similarity_megfile(
                gallery_dir=eval_dir,
                probe_image=probe_image,
                backend=backend_name,
                output_json=None,     # 不为每个 probe 单独写 JSON
                processor=processor,  # 复用本进程已加载的 backend
                model=model,
                verbose=False,
            )
        except Exception as e:
            print(f"[ERROR] {model_dir}: probe={probe_image} 计算相似度失败 ----> {e}")
            continue

        mean_sim = result.get("mean_similarity", None)
        if mean_sim is None:
            print(f"[WARN] {model_dir}: probe={probe_image} 结果中没有 mean_similarity 字段，跳过该 probe")
            continue

        per_probe_mean[probe_image] = mean_sim
        total_mean += mean_sim
        valid_probes += 1
        print(f"[OK] {model_dir}: probe={probe_image}, mean_similarity={mean_sim:.4f}")

    if valid_probes == 0:
        print(f"[WARN] {model_dir}: 所有 probe 计算均失败，跳过写入 JSON")
        return

    overall_mean = total_mean / valid_probes

    final_result = {
        "backend": backend_name,
        # 兼容原字段 + 新增 probe_mode/probe_dir
        "probe_mode": probe_mode,          # "demo" or "content"
        "probe_dir": probe_dir,            # 实际使用的 probe 目录
        "demo_dir": demo_dir,              # demo_images 目录（即使 probe_mode=content 也保留路径）
        "content_dir": content_dir,        # content_100 目录路径
        "eval_dir": eval_dir,
        "content_dir_name": content_dir_name,
        "num_probe_images": valid_probes,
        "per_probe_mean_similarity": per_probe_mean,
        "overall_mean_similarity": overall_mean,
    }

    out_dir = os.path.dirname(out_json)
    if out_dir:
        smart_makedirs(out_dir, exist_ok=True)
    with mopen(out_json, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] {model_dir}: {valid_probes} 个 probe 的 mean_similarity 平均值 = "
        f"{overall_mean:.4f}，已写入 {out_json}（probe_mode={probe_mode}）"
    )


def worker_main(
    worker_id: int,
    model_dirs: List[str],
    backend: str,
    output_name: str,
    overwrite: bool,
    gpu_id: int = None,
    probe_mode: str = "demo",
    content_dir_name: str = "content_100",
) -> None:
    """
    单个 worker 进程：
    - 绑定到一个 GPU（如果提供 gpu_id 且 cuda 可用）
    - 设置 content_score.device
    - load_backend
    - 依次处理分配到的 model_dirs
    """
    if not model_dirs:
        print(f"[WORKER-{worker_id}] 没有分配到任何子目录，直接退出")
        return

    # 设置设备
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[WORKER-{worker_id}] 使用 GPU {gpu_id}，device = {device}")
    else:
        device = torch.device("cpu")
        print(f"[WORKER-{worker_id}] 未使用 GPU，device = CPU")

    # 覆盖 content_score 中的全局 device，让后续 load_backend / extract_feature 都用这个设备
    content_score.device = device

    # 每个进程各自加载一份模型到自己的 device 上
    backend_name, processor, model = content_score.load_backend(backend)
    print(
        f"[WORKER-{worker_id}] 已加载后端 {backend_name}（probe_mode={probe_mode}），"
        f"待处理子目录数 = {len(model_dirs)}"
    )

    for model_dir in model_dirs:
        try:
            print(f"[WORKER-{worker_id}] 开始处理 model_dir = {model_dir}")
            process_single_model_dir(
                model_dir=model_dir,
                backend_name=backend_name,
                processor=processor,
                model=model,
                output_name=output_name,
                overwrite=overwrite,
                probe_mode=probe_mode,
                content_dir_name=content_dir_name,
            )
        except Exception as e:
            print(f"[WORKER-{worker_id}] [ERROR] 处理 {model_dir} 失败: {e}")


# ==================== 主逻辑 ====================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "多进程：遍历 root 下每个子目录，"
            "使用 demo_images 或 content_100 中的每一张图作为 probe，"
            "分别对 eval_images 计算 mean_similarity，"
            "再对所有 probe 的 mean_similarity 取平均，"
            "结果写入子目录下的 content_similarity.json（或自定义文件名）"
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="包含多个 model 子目录的根目录（本地或 s3:// 等桶路径）",
    )
    parser.add_argument(
        "--backend",
        choices=["dino", "clip"],
        default="dino",
        help="内容特征后端，传给 compute_gallery_similarity_megfile，默认 dino",
    )
    parser.add_argument(
        "--output-name",
        default="content_similarity.json",
        help="输出 JSON 文件名，默认 content_similarity.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如指定，则即使 content_similarity.json 已存在也会覆盖重算",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行进程数（建议不超过 GPU 数量；CPU 跑也可以）",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help=(
            "可选，逗号分隔的 GPU 编号，例如 '0,1,2'。"
            "若不指定则默认使用所有可见 GPU 并按 worker 轮询分配；"
            "若无 GPU 或不想用 GPU，可以不传并把 CUDA_VISIBLE_DEVICES 置空。"
        ),
    )
    parser.add_argument(
        "--probe-mode",
        choices=["demo", "content"],
        default="demo",
        help=(
            "probe 来源："
            "demo  -> 使用 demo_images 中的图片作为 probe（默认）；"
            "content -> 使用 content_100（或 --content-dir-name 指定）中的图片作为 probe。"
        ),
    )
    parser.add_argument(
        "--content-dir-name",
        type=str,
        default="content_100",
        help="当 --probe-mode=content 时，作为 probe 的子目录名，默认 content_100",
    )

    args = parser.parse_args()

    root = args.root.rstrip("/")

    # 用 content_score 里已经写好的工具枚举 model 目录（支持本地 / 桶）
    model_dirs = content_score.iter_model_dirs_megfile(root)
    if not model_dirs:
        raise SystemExit(f"在 root={root} 下没有找到任何子目录")

    print(
        f"[INFO] 在 {root} 下找到 {len(model_dirs)} 个子目录，将进行内容相似度评估 "
        f"(probe_mode={args.probe_mode})"
    )

    # 计算实际 worker 数
    num_workers = max(1, args.num_workers)
    num_workers = min(num_workers, len(model_dirs))

    # 解析 GPU 列表
    if torch.cuda.is_available():
        detected_gpus = torch.cuda.device_count()
    else:
        detected_gpus = 0

    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(detected_gpus))

    if detected_gpus == 0 or not gpu_ids:
        print("[INFO] 未检测到可用 GPU 或未指定 gpu-ids，将在 CPU 上运行所有进程")
        gpu_ids = [None] * num_workers  # 每个 worker 的 gpu_id 都是 None
    else:
        print(f"[INFO] 可用 GPU: {gpu_ids}（检测到 {detected_gpus} 张卡）")

    # 把 model_dirs 均匀分配给 num_workers 个进程
    chunks = split_into_chunks(model_dirs, num_workers)
    num_workers = len(chunks)  # 可能少于原 num_workers（当 model_dirs < num_workers 时）

    print(f"[INFO] 实际启动 {num_workers} 个 worker 进程")

    processes: List[mp.Process] = []
    for i in range(num_workers):
        dirs_i = chunks[i]
        # 为该 worker 选一个 gpu_id（若 GPU 数 < worker 数，就循环使用）
        if gpu_ids[0] is None:
            gpu_id = None
        else:
            gpu_id = gpu_ids[i % len(gpu_ids)]

        p = mp.Process(
            target=worker_main,
            args=(
                i,                 # worker_id
                dirs_i,            # 分配到的 model_dirs
                args.backend,      # backend
                args.output_name,  # 输出文件名
                args.overwrite,    # 是否覆写
                gpu_id,            # 分配的 GPU
                args.probe_mode,   # probe 模式
                args.content_dir_name,  # content_100 子目录名
            ),
        )
        p.start()
        processes.append(p)

    # 等所有 worker 结束
    for p in processes:
        p.join()

    print("[DONE] 所有子目录处理完成。")


if __name__ == "__main__":
    # 为了 CUDA + 多进程更稳，建议使用 spawn
    mp.set_start_method("spawn", force=True)
    main()

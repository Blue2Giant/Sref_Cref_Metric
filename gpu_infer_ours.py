#!/usr/bin/env python3
"""Timed multi-GPU stress test with memory reservation and continuous compute.

Designed for short-lived validation / burn-in style checks:
- Detects visible GPUs automatically
- Reserves roughly the requested amount of VRAM on each selected GPU
- Runs continuous matmul and/or conv compute to keep utilization and power up
- Prints live NVML stats and exits automatically after the requested duration

/data/benchmark_metrics/run_gpu_infer.sh \
  --minutes 9999999999999 \
  --memory-gb 20 \
  --compute-mode matmul \
  --matmul-size 8384 \
  --matmul-iters 4 \
  --stats-interval 2

"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pynvml
import torch
import torch.nn.functional as F


BYTES_PER_GIB = 1024 ** 3
MIN_ALLOC_CHUNK_BYTES = 64 * 1024 ** 2


@dataclass(frozen=True)
class StressConfig:
    deadline_ts: float
    memory_gb: float
    reserve_gb: float
    chunk_gb: float
    matmul_size: int
    matmul_iters: int
    conv_batch: int
    conv_channels: int
    conv_image_size: int
    conv_iters: int
    compute_mode: str
    dtype_name: str
    touch_filler_every: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timed multi-GPU stress test with VRAM reservation and sustained compute."
    )
    parser.add_argument("--minutes", type=float, default=10.0, help="Run time in minutes.")
    parser.add_argument(
        "--memory-gb",
        type=float,
        default=20.0,
        help="Approximate VRAM to occupy per GPU in GiB.",
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=2.0,
        help="Leave at least this much free memory per GPU to avoid OOM from temporary workspaces.",
    )
    parser.add_argument(
        "--chunk-gb",
        type=float,
        default=0.5,
        help="Chunk size in GiB for filler allocations.",
    )
    parser.add_argument(
        "--gpus",
        default="all",
        help='Visible GPU ids to use, e.g. "0,1,3". Default: all visible GPUs.',
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Compute dtype. Default picks bf16 when supported, else fp16.",
    )
    parser.add_argument(
        "--compute-mode",
        choices=("mixed", "matmul", "conv"),
        default="mixed",
        help="Compute kernel mix. mixed = matmul plus periodic conv.",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=12288,
        help="Square matrix size used by matmul kernels.",
    )
    parser.add_argument(
        "--matmul-iters",
        type=int,
        default=6,
        help="Number of matmul iterations per compute cycle.",
    )
    parser.add_argument(
        "--conv-batch",
        type=int,
        default=8,
        help="Batch size for conv kernels when conv is enabled.",
    )
    parser.add_argument(
        "--conv-channels",
        type=int,
        default=192,
        help="Channel count for conv kernels when conv is enabled.",
    )
    parser.add_argument(
        "--conv-image-size",
        type=int,
        default=96,
        help="Input image width/height for conv kernels when conv is enabled.",
    )
    parser.add_argument(
        "--conv-iters",
        type=int,
        default=2,
        help="Number of conv iterations per compute cycle.",
    )
    parser.add_argument(
        "--touch-filler-every",
        type=int,
        default=8,
        help="Touch one filler chunk every N compute cycles. 0 disables this.",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=5.0,
        help="How often to print NVML stats in seconds.",
    )
    return parser.parse_args()


def format_gib(num_bytes: int | float) -> str:
    return f"{num_bytes / BYTES_PER_GIB:.2f}GiB"


def pick_dtype(dtype_name: str, device_idx: int) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported(device_idx) else torch.float16


def parse_visible_gpu_ids(spec: str) -> list[int]:
    visible_count = torch.cuda.device_count()
    if visible_count == 0:
        raise RuntimeError("No visible CUDA devices.")
    if spec.strip().lower() == "all":
        return list(range(visible_count))

    requested = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        gpu_idx = int(part)
        if gpu_idx < 0 or gpu_idx >= visible_count:
            raise ValueError(f"GPU id {gpu_idx} is out of range for {visible_count} visible devices.")
        requested.append(gpu_idx)

    if not requested:
        raise ValueError("No GPU ids parsed from --gpus.")
    return sorted(set(requested))


def logical_to_physical_map() -> dict[int, int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        visible_count = torch.cuda.device_count()
        return {idx: idx for idx in range(visible_count)}

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    mapping: dict[int, int] = {}
    for logical_idx, part in enumerate(parts):
        try:
            mapping[logical_idx] = int(part)
        except ValueError:
            # UUID-based mapping is harder to resolve robustly; fall back to identity.
            mapping[logical_idx] = logical_idx
    return mapping


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def allocate_filler(device: torch.device, target_bytes: int, chunk_bytes: int) -> tuple[list[torch.Tensor], int]:
    filler_blocks: list[torch.Tensor] = []
    allocated_bytes = 0
    remaining = max(0, target_bytes)
    current_chunk = max(MIN_ALLOC_CHUNK_BYTES, chunk_bytes)

    while remaining > 0:
        this_chunk = min(current_chunk, remaining)
        try:
            block = torch.empty(this_chunk, dtype=torch.uint8, device=device)
            block.fill_(1)
            filler_blocks.append(block)
            allocated_bytes += this_chunk
            remaining -= this_chunk
        except RuntimeError:
            if this_chunk <= MIN_ALLOC_CHUNK_BYTES:
                break
            current_chunk = max(MIN_ALLOC_CHUNK_BYTES, this_chunk // 2)

    return filler_blocks, allocated_bytes


def build_compute_tensors(device_idx: int, config: StressConfig) -> tuple[list[torch.Tensor], dict[str, object]]:
    dtype = pick_dtype(config.dtype_name, device_idx)
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device_idx)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    compute_tensors: list[torch.Tensor] = []
    state: dict[str, object] = {
        "device": device,
        "dtype": dtype,
    }

    if config.compute_mode in {"mixed", "matmul"}:
        size = config.matmul_size
        mat_a = torch.randn((size, size), device=device, dtype=dtype)
        mat_b = torch.randn((size, size), device=device, dtype=dtype)
        mat_c = torch.empty((size, size), device=device, dtype=dtype)
        compute_tensors.extend([mat_a, mat_b, mat_c])
        state["matmul"] = [mat_a, mat_b, mat_c]

    if config.compute_mode in {"mixed", "conv"}:
        conv_x = torch.randn(
            (config.conv_batch, config.conv_channels, config.conv_image_size, config.conv_image_size),
            device=device,
            dtype=dtype,
        )
        conv_w = torch.randn(
            (config.conv_channels, config.conv_channels, 3, 3),
            device=device,
            dtype=dtype,
        )
        compute_tensors.extend([conv_x, conv_w])
        state["conv"] = [conv_x, conv_w]

    torch.cuda.synchronize(device)
    state["compute_bytes"] = sum(tensor_nbytes(t) for t in compute_tensors)
    return compute_tensors, state


def warmup(device: torch.device, state: dict[str, object], config: StressConfig) -> None:
    with torch.no_grad():
        if "matmul" in state:
            mat_a, mat_b, mat_c = state["matmul"]  # type: ignore[misc]
            torch.matmul(mat_a, mat_b, out=mat_c)
            torch.matmul(mat_c, mat_a, out=mat_b)
        if "conv" in state:
            conv_x, conv_w = state["conv"]  # type: ignore[misc]
            conv_y = F.conv2d(conv_x, conv_w, padding=1)
            state["conv"] = [conv_y, conv_w]
    torch.cuda.synchronize(device)


def compute_cycle(state: dict[str, object], config: StressConfig, cycle_idx: int) -> None:
    with torch.no_grad():
        if "matmul" in state:
            mat_a, mat_b, mat_c = state["matmul"]  # type: ignore[misc]
            for _ in range(config.matmul_iters):
                torch.matmul(mat_a, mat_b, out=mat_c)
                mat_a, mat_b, mat_c = mat_b, mat_c, mat_a
            state["matmul"] = [mat_a, mat_b, mat_c]

        if "conv" in state:
            conv_x, conv_w = state["conv"]  # type: ignore[misc]
            for _ in range(config.conv_iters):
                conv_x = F.conv2d(conv_x, conv_w, padding=1)
            state["conv"] = [conv_x, conv_w]

        if config.touch_filler_every > 0 and cycle_idx % config.touch_filler_every == 0:
            filler_blocks = state.get("filler_blocks", [])
            if filler_blocks:
                block = filler_blocks[cycle_idx % len(filler_blocks)]
                block[: min(block.numel(), 8 * 1024 * 1024)].add_(1)


def worker_main(
    logical_gpu_idx: int,
    physical_gpu_idx: int,
    config: StressConfig,
    status_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    try:
        torch.cuda.set_device(logical_gpu_idx)
        device = torch.device(f"cuda:{logical_gpu_idx}")
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        requested_bytes = int(config.memory_gb * BYTES_PER_GIB)
        reserve_bytes = int(config.reserve_gb * BYTES_PER_GIB)
        max_safe_target = max(0, int(free_bytes) - reserve_bytes)
        effective_target = min(requested_bytes, max_safe_target)

        compute_tensors, state = build_compute_tensors(logical_gpu_idx, config)
        compute_bytes = int(state["compute_bytes"])
        filler_target = max(0, effective_target - compute_bytes)
        filler_blocks, filler_bytes = allocate_filler(
            device=device,
            target_bytes=filler_target,
            chunk_bytes=int(config.chunk_gb * BYTES_PER_GIB),
        )
        state["filler_blocks"] = filler_blocks
        state["compute_tensors"] = compute_tensors

        warmup(device, state, config)
        allocated_bytes = compute_bytes + filler_bytes
        status_queue.put(
            {
                "gpu": logical_gpu_idx,
                "physical_gpu": physical_gpu_idx,
                "status": "ready",
                "requested_bytes": requested_bytes,
                "effective_target": effective_target,
                "allocated_bytes": allocated_bytes,
                "compute_bytes": compute_bytes,
                "filler_bytes": filler_bytes,
                "dtype": str(state["dtype"]).replace("torch.", ""),
            }
        )

        cycle_idx = 0
        while time.time() < config.deadline_ts and not stop_event.is_set():
            compute_cycle(state, config, cycle_idx)
            cycle_idx += 1
        torch.cuda.synchronize(device)
    except Exception as exc:  # pragma: no cover - surfaced to parent process
        status_queue.put(
            {
                "gpu": logical_gpu_idx,
                "physical_gpu": physical_gpu_idx,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise


def monitor_loop(
    logical_gpu_ids: Iterable[int],
    logical_to_physical: dict[int, int],
    deadline_ts: float,
    stats_interval: float,
    stop_event: mp.Event,
) -> tuple[dict[int, list[float]], dict[int, float]]:
    handles = {
        gpu: pynvml.nvmlDeviceGetHandleByIndex(logical_to_physical[gpu])
        for gpu in logical_gpu_ids
    }
    power_samples_w: dict[int, list[float]] = {gpu: [] for gpu in logical_gpu_ids}
    peak_power_w: dict[int, float] = {gpu: 0.0 for gpu in logical_gpu_ids}

    while time.time() < deadline_ts and not stop_event.is_set():
        parts = []
        total_power = 0.0
        for logical_gpu_idx in logical_gpu_ids:
            handle = handles[logical_gpu_idx]
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            total_power += power_w
            power_samples_w[logical_gpu_idx].append(power_w)
            peak_power_w[logical_gpu_idx] = max(peak_power_w[logical_gpu_idx], power_w)
            parts.append(
                (
                    f"GPU{logical_gpu_idx}"
                    f"(phys{logical_to_physical[logical_gpu_idx]}) "
                    f"util={util.gpu:>3d}% mem={mem.used / BYTES_PER_GIB:>5.1f}GiB "
                    f"power={power_w:>6.1f}W temp={temp:>2d}C"
                )
            )

        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] total_power={total_power:.1f}W | " + " | ".join(parts), flush=True)
        time.sleep(stats_interval)

    return power_samples_w, peak_power_w


def print_startup_summary(args: argparse.Namespace, logical_gpu_ids: list[int], logical_to_physical: dict[int, int]) -> None:
    duration_seconds = args.minutes * 60.0
    mapping = ", ".join(f"logical {gpu} -> physical {logical_to_physical[gpu]}" for gpu in logical_gpu_ids)
    print(
        "Starting timed GPU stress test "
        f"for {duration_seconds:.1f}s on {len(logical_gpu_ids)} GPU(s).",
        flush=True,
    )
    print(
        (
            f"Requested per-GPU VRAM target: {args.memory_gb:.2f}GiB, "
            f"reserve headroom: {args.reserve_gb:.2f}GiB, compute mode: {args.compute_mode}, "
            f"dtype: {args.dtype}, matmul_size: {args.matmul_size}."
        ),
        flush=True,
    )
    print(f"GPU mapping: {mapping}", flush=True)


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available in the current Python environment.", file=sys.stderr)
        return 1
    if args.minutes <= 0:
        print("--minutes must be > 0.", file=sys.stderr)
        return 1
    if args.memory_gb <= 0:
        print("--memory-gb must be > 0.", file=sys.stderr)
        return 1
    if args.chunk_gb <= 0:
        print("--chunk-gb must be > 0.", file=sys.stderr)
        return 1

    logical_gpu_ids = parse_visible_gpu_ids(args.gpus)
    mapping = logical_to_physical_map()
    print_startup_summary(args, logical_gpu_ids, mapping)

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        print(f"Failed to initialize NVML: {exc}", file=sys.stderr)
        return 1

    deadline_ts = time.time() + args.minutes * 60.0
    config = StressConfig(
        deadline_ts=deadline_ts,
        memory_gb=args.memory_gb,
        reserve_gb=args.reserve_gb,
        chunk_gb=args.chunk_gb,
        matmul_size=args.matmul_size,
        matmul_iters=args.matmul_iters,
        conv_batch=args.conv_batch,
        conv_channels=args.conv_channels,
        conv_image_size=args.conv_image_size,
        conv_iters=args.conv_iters,
        compute_mode=args.compute_mode,
        dtype_name=args.dtype,
        touch_filler_every=args.touch_filler_every,
    )

    ctx = mp.get_context("spawn")
    status_queue: mp.Queue = ctx.Queue()
    stop_event = ctx.Event()

    def handle_signal(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}, stopping workers...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    workers: list[mp.Process] = []
    for logical_gpu_idx in logical_gpu_ids:
        process = ctx.Process(
            target=worker_main,
            args=(logical_gpu_idx, mapping[logical_gpu_idx], config, status_queue, stop_event),
            daemon=False,
        )
        process.start()
        workers.append(process)

    ready_gpus: set[int] = set()
    try:
        while len(ready_gpus) < len(workers):
            message = status_queue.get(timeout=120.0)
            if message["status"] == "error":
                stop_event.set()
                print(
                    f"Worker failed on GPU{message['gpu']} (phys{message['physical_gpu']}): {message['error']}",
                    file=sys.stderr,
                    flush=True,
                )
                for process in workers:
                    process.join(timeout=2.0)
                    if process.is_alive():
                        process.terminate()
                pynvml.nvmlShutdown()
                return 1

            ready_gpus.add(message["gpu"])
            print(
                (
                    f"GPU{message['gpu']} (phys{message['physical_gpu']}) ready: "
                    f"allocated {format_gib(message['allocated_bytes'])} total "
                    f"(compute {format_gib(message['compute_bytes'])}, filler {format_gib(message['filler_bytes'])}), "
                    f"target {format_gib(message['effective_target'])}, dtype={message['dtype']}."
                ),
                flush=True,
            )
    except queue.Empty:
        stop_event.set()
        print("Timed out waiting for worker startup.", file=sys.stderr, flush=True)
        for process in workers:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
        pynvml.nvmlShutdown()
        return 1

    power_samples_w: dict[int, list[float]]
    peak_power_w: dict[int, float]
    try:
        power_samples_w, peak_power_w = monitor_loop(
            logical_gpu_ids=logical_gpu_ids,
            logical_to_physical=mapping,
            deadline_ts=deadline_ts,
            stats_interval=args.stats_interval,
            stop_event=stop_event,
        )
    finally:
        stop_event.set()
        for process in workers:
            process.join(timeout=30.0)
            if process.is_alive():
                process.terminate()

    exit_code = 0
    for process in workers:
        if process.exitcode not in (0, None):
            exit_code = 1

    print("Run finished. Power summary:", flush=True)
    for logical_gpu_idx in logical_gpu_ids:
        samples = power_samples_w.get(logical_gpu_idx, [])
        avg_power = sum(samples) / len(samples) if samples else math.nan
        peak_power = peak_power_w.get(logical_gpu_idx, math.nan)
        print(
            f"GPU{logical_gpu_idx} (phys{mapping[logical_gpu_idx]}): "
            f"avg_power={avg_power:.1f}W peak_power={peak_power:.1f}W samples={len(samples)}",
            flush=True,
        )

    pynvml.nvmlShutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

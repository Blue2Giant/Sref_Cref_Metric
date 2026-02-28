"""并发执行工具"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from loguru import logger
from tqdm import tqdm

T = TypeVar("T")  # 输入类型
R = TypeVar("R")  # 结果类型


@dataclass
class Task(Generic[T]):
    """
    任务封装：分离处理数据和上下文信息

    Attributes:
        input_data: 传给处理函数的数据
        context: 存储/回调时需要的额外信息（不传给处理函数）
    """

    input_data: T
    context: dict = field(default_factory=dict)


@dataclass
class TaskResult(Generic[R]):
    """
    任务结果

    Attributes:
        result: 处理函数返回值
        context: 原始任务的 context（透传）
        success: 是否成功
        error: 失败时的异常
    """

    result: R | None = None
    context: dict = field(default_factory=dict)
    success: bool = True
    error: Exception | None = None


def run_parallel(
    tasks: list[Task[T]],
    process_func: Callable[[T], R],
    max_workers: int = 32,
    desc: str = "Processing",
) -> list[TaskResult[R]]:
    """
    并行执行任务（简单版）

    Args:
        tasks: Task 列表，每个 Task 包含 input_data 和 context
        process_func: 处理函数，签名为 (input_data: T) -> R
        max_workers: 最大线程数
        desc: 进度条描述

    Returns:
        TaskResult 列表，每个结果包含 result 和原始 context

    Example:
        # 定义任务
        tasks = [
            Task(input_data=image_bytes, context={"image_id": id1, "seq_id": seq1}),
            Task(input_data=image_bytes, context={"image_id": id2, "seq_id": seq2}),
        ]

        # 定义处理函数（只需要 input_data）
        def annotate(image_bytes: bytes) -> dict:
            return call_vlm(image_bytes, prompt)

        # 执行
        results = run_parallel(tasks, annotate, max_workers=32)

        # 结果包含 context，方便存储
        for r in results:
            if r.success:
                save(r.result, r.context["image_id"], r.context["seq_id"])
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_func, task.input_data): task for task in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=desc):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(
                    TaskResult(result=result, context=task.context, success=True)
                )
            except Exception as e:
                logger.error(f"任务失败: context={task.context}, 错误: {e}")
                results.append(
                    TaskResult(
                        result=None, context=task.context, success=False, error=e
                    )
                )

    return results


def run_producer_consumer(
    tasks: list[Task[T]],
    process_func: Callable[[T], R],
    save_func: Callable[[list[TaskResult[R]]], None],
    max_workers: int = 64,
    batch_size: int = 32,
    save_errors: bool = False,
):
    """
    生产者-消费者模式执行

    Args:
        tasks: Task 列表
        process_func: 处理函数，签名为 (input_data: T) -> R
        save_func: 批量保存函数，签名为 (results: list[TaskResult]) -> None
                   每个 TaskResult 包含 result 和 context
        max_workers: 工作线程数
        batch_size: 批量保存大小
        save_errors: 是否保存失败的任务

    Example:
        # 定义任务
        tasks = [
            Task(
                input_data={"image": img_bytes, "prompt": prompt},
                context={"image_id": id1, "seq_id": seq1, "annotation_name": "tag"}
            ),
            ...
        ]

        # 处理函数：只关心业务逻辑
        def annotate(data: dict) -> dict:
            return call_vlm(data["image"], data["prompt"])

        # 保存函数：使用 context 构建存储结构
        def save_batch(results: list[TaskResult]):
            annotations = []
            for r in results:
                if r.success:
                    annotations.append({
                        "name": r.context["annotation_name"],
                        "value_json": r.result,
                        "sequence_id": r.context["seq_id"],
                        "participants": [(r.context["image_id"], "image", "target")]
                    })
            save_annotations(storager, annotations)

        # 执行
        run_producer_consumer(tasks, annotate, save_batch, max_workers=128)
    """
    results_queue = queue.Queue(maxsize=batch_size * 4)
    stop_event = threading.Event()

    def producer():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_func, task.input_data): task for task in tasks
            }

            for future in tqdm(
                as_completed(future_to_task), total=len(tasks), desc="Processing"
            ):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results_queue.put(
                        TaskResult(result=result, context=task.context, success=True)
                    )
                except Exception as e:
                    logger.error(f"任务失败: context={task.context}, 错误: {e}")
                    if save_errors:
                        results_queue.put(
                            TaskResult(context=task.context, success=False, error=e)
                        )

        stop_event.set()

    def consumer():
        buffer = []
        while not stop_event.is_set() or not results_queue.empty():
            try:
                result = results_queue.get(timeout=1.0)
                buffer.append(result)

                if len(buffer) >= batch_size:
                    save_func(buffer)
                    buffer.clear()
            except queue.Empty:
                continue

        if buffer:
            save_func(buffer)

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

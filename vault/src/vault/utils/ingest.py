import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Optional

import megfile
from loguru import logger
from tqdm import tqdm


# --- 新增: 用于在进程间传递结构化结果 ---
@dataclass
class WorkerResult:
    """封装工作进程的执行结果，用于进程间通信。"""

    success: bool
    task_id: int
    result: Optional[Any] = None
    exception: Optional[Exception] = None
    traceback: Optional[str] = None


# --- 内部使用的包装函数，不要直接调用 ---
def _worker_wrapper(func: Callable, task: Any, task_id: int, *args) -> WorkerResult:
    """
    在子进程中执行实际工作函数并捕获其结果或异常。
    """
    try:
        result = func(task, task_id, *args)
        return WorkerResult(success=True, task_id=task_id, result=result)
    except Exception as e:
        tb_str = traceback.format_exc()
        return WorkerResult(
            success=False, task_id=task_id, exception=e, traceback=tb_str
        )


def run_concurrently(
    worker_func: Callable,
    tasks: list[Any],
    num_workers: int,
    *args,
    task_timeout: Optional[int] = None,
):
    """
    使用进程池并发执行任务，并提供健壮的错误处理和内存隔离。

    Args:
        worker_func (Callable): 在工作进程中执行的函数。
                               签名应为: `worker_func(task, task_id, *args)`
        tasks (list[Any]): 任务列表，每个元素都是 worker_func 的一个输入。
        num_workers (int): 要启动的工作进程数量。
        *args: 传递给 worker_func 的额外位置参数。
        task_timeout (Optional[int]): 每个任务的超时时间（秒）。如果任务超时，
                                      它将被取消。默认为 None。
    """
    start_time = time.time()

    if not tasks:
        logger.info("任务列表为空，无需执行。")
        return

    # 调整工作进程数以避免资源浪费
    effective_workers = min(min(num_workers, len(tasks)), os.cpu_count() or 32)
    logger.info(f"启动 {effective_workers} 个工作进程处理 {len(tasks)} 个任务 ")

    success_count = 0
    fail_count = 0

    try:
        with ProcessPoolExecutor(
            max_workers=effective_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(_worker_wrapper, worker_func, task, i, *args): i
                for i, task in enumerate(tasks)
            }

            with tqdm(
                total=len(tasks),
                desc=f"并发处理中 (Workers: {effective_workers})",
                smoothing=0.1,
            ) as pbar:
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        # 从 future 中获取 WorkerResult 对象，并设置超时
                        worker_result: WorkerResult = future.result(
                            timeout=task_timeout
                        )

                        if worker_result.success:
                            success_count += 1
                            result_str = (
                                f" -> {worker_result.result}"
                                if worker_result.result is not None
                                else ""
                            )
                            logger.success(
                                f"任务 chunk-{worker_result.task_id} 成功完成。{result_str}"
                            )
                        else:
                            fail_count += 1
                            # 清晰地打印捕获到的异常和追溯信息
                            logger.error(
                                f"任务 chunk-{worker_result.task_id} 在子进程中失败！"
                            )
                            logger.error(f"异常: {worker_result.exception}")
                            logger.error(
                                f"子进程追溯信息:\n--- START TRACEBACK ---\n{worker_result.traceback}\n--- END TRACEBACK ---"
                            )

                    except TimeoutError:
                        fail_count += 1
                        logger.warning(
                            f"任务 chunk-{task_id} 超时 ({task_timeout}s)，已被取消。"
                        )
                        future.cancel()  # 尝试取消任务
                    except Exception as e:
                        # 这里的异常通常是进程池本身的问题，比如 BrokenProcessPool
                        fail_count += 1
                        logger.critical(f"处理任务 chunk-{task_id} 时发生严重错误: {e}")
                        logger.error(f"{traceback.format_exc()}")

                    pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("\n检测到 Ctrl+C！正在等待进程关闭...")
    finally:
        total_time = timedelta(seconds=time.time() - start_time)
        logger.info("-" * 50)
        logger.info("运行摘要:")
        logger.info(f"  - 成功任务: {success_count}")
        logger.info(f"  - 失败任务: {fail_count}")
        logger.info(f"  - 总耗时: {total_time}")
        logger.info("-" * 50)


def expand_tar_urls(roots: str | list[str]) -> list[str]:
    if isinstance(roots, str):
        roots = [roots]

    _urls = []

    for root in roots:
        if "*" in root:
            # 如果根目录路径中包含通配符 *，则使用 megfile.smart_glob 函数查找符合通配符模式的所有文件
            urls = megfile.smart_glob(root)
        elif root.endswith(".tar") or root.endswith(".tar.gz"):
            urls = [root]
        else:
            urls = megfile.smart_glob(
                megfile.smart_path_join(root, "**/*.tar")
            ) + megfile.smart_glob(megfile.smart_path_join(root, "**/*.tar.gz"))

        _urls.extend(urls)
    return _urls

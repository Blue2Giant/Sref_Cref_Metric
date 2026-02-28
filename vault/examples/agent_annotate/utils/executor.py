"""
Elegant concurrent executor based on concurrent.futures.

Core design:
- Main thread handles everything (no separate producer/consumer threads)
- Uses ThreadPoolExecutor context manager for automatic cleanup
- Uses as_completed for non-blocking result processing
- Graceful interrupt handling with double Ctrl+C mechanism
- Clear function names over complex documentation
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar
import time
import signal

from loguru import logger
from tqdm import tqdm

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


@dataclass
class Task(Generic[T]):
    """
    Task wrapper: separates processing data from context.

    Attributes:
        input_data: Data passed to process function
        context: Extra info for saving (not passed to process function)
    """

    input_data: T
    context: dict = field(default_factory=dict)


@dataclass
class TaskResult(Generic[R]):
    """
    Task execution result.

    Attributes:
        result: Process function return value
        context: Original task context (passthrough)
        success: Whether execution succeeded
        error: Exception if failed
    """

    result: R | None = None
    context: dict = field(default_factory=dict)
    success: bool = True
    error: Exception | None = None


@dataclass
class Stats:
    """
    Execution statistics with computed properties.
    """

    total: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def items_per_second(self) -> float:
        return self.processed / self.elapsed_seconds if self.elapsed_seconds > 0 else 0

    @property
    def eta_seconds(self) -> float:
        remaining = self.total - self.processed
        return remaining / self.items_per_second if self.items_per_second > 0 else 0

    @property
    def progress_percent(self) -> float:
        return (self.processed / self.total * 100) if self.total > 0 else 0

    def log_progress(self):
        """Log current progress with ETA."""
        logger.info(
            f"Progress: {self.processed}/{self.total} ({self.progress_percent:.1f}%) | "
            f"Success: {self.success} | Failed: {self.failed} | "
            f"Speed: {self.items_per_second:.2f} items/s | "
            f"ETA: {self.eta_seconds/60:.1f} min"
        )

    def log_summary(self):
        """Log final execution summary."""
        logger.info("=" * 70)
        logger.info("Execution Summary:")
        logger.info(f"  Total tasks:    {self.total}")
        logger.info(f"  Processed:      {self.processed}")
        logger.info(f"  Success:        {self.success}")
        logger.info(f"  Failed:         {self.failed}")
        logger.info(f"  Total time:     {self.elapsed_seconds/60:.2f} min")
        logger.info(f"  Average speed:  {self.items_per_second:.2f} items/s")
        logger.info("=" * 70)


class BatchExecutor:
    """
    Concurrent batch executor using ThreadPoolExecutor.

    Key features:
    - Main thread handles all coordination (no separate threads)
    - Non-blocking: processes results as they complete
    - Graceful interrupt: double Ctrl+C mechanism
    - Batch saving: accumulates results and saves in batches
    """

    def __init__(
        self,
        max_workers: int = 128,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        log_every_n_items: int = 50,
    ):
        """
        Args:
            max_workers: Maximum concurrent threads
            batch_size: Number of results to accumulate before saving
            show_progress_bar: Whether to show tqdm progress bar
            log_every_n_items: Log progress every N items
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.log_every_n_items = log_every_n_items

        self.interrupted = False
        self.stats = Stats()

    def _setup_interrupt_handler(self):
        """
        Setup graceful interrupt handler.

        First Ctrl+C: graceful shutdown (cancel remaining tasks)
        Second Ctrl+C: force exit
        """
        original_handler = signal.getsignal(signal.SIGINT)

        def handle_interrupt(signum, frame):
            if not self.interrupted:
                logger.warning(
                    "\nReceived interrupt signal (Ctrl+C), gracefully stopping..."
                )
                logger.warning("Press Ctrl+C again to force exit")
                self.interrupted = True
            else:
                logger.error("Force exit!")
                signal.signal(signal.SIGINT, original_handler)
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handle_interrupt)
        return original_handler

    def _cancel_remaining_tasks(self, future_to_task: dict):
        """Cancel all unfinished tasks."""
        for future in future_to_task:
            if not future.done():
                future.cancel()

    def _process_completed_future(
        self, future, task: Task, save_errors: bool
    ) -> TaskResult | None:
        """
        Process a completed future and return TaskResult.

        Returns None if task failed and save_errors=False.
        """
        try:
            result = future.result()
            self.stats.success += 1
            return TaskResult(result=result, context=task.context, success=True)

        except Exception as e:
            logger.error(f"Task failed: context={task.context}, error: {e}")
            self.stats.failed += 1

            if save_errors:
                return TaskResult(
                    result=None, context=task.context, success=False, error=e
                )
            return None

    def _save_batch_if_ready(self, buffer: list, save_func: Callable):
        """Save batch if buffer is full, then clear buffer."""
        if len(buffer) >= self.batch_size:
            try:
                save_func(buffer)
            except Exception as e:
                logger.error(f"Save batch failed: {e}")
            finally:
                buffer.clear()

    def _save_remaining_results(self, buffer: list, save_func: Callable):
        """Save remaining results in buffer."""
        if buffer:
            logger.info(f"Saving remaining {len(buffer)} results...")
            try:
                save_func(buffer)
            except Exception as e:
                logger.error(f"Save remaining results failed: {e}")

    def execute(
        self,
        tasks: list[Task[T]],
        process_func: Callable[[T], R],
        save_func: Callable[[list[TaskResult[R]]], None],
        save_errors: bool = False,
    ) -> Stats:
        """
        Execute tasks concurrently with batch saving.

        Process flow:
        1. Submit all tasks to ThreadPoolExecutor at once
        2. Process results as they complete (non-blocking, no ordering)
        3. Accumulate results in buffer
        4. Save batch when buffer reaches batch_size
        5. Handle interrupts gracefully

        Args:
            tasks: List of Task objects
            process_func: Function to process input_data, signature: (T) -> R
            save_func: Function to save batch, signature: (list[TaskResult]) -> None
            save_errors: Whether to save failed task results

        Returns:
            Execution statistics

        Example:
            executor = BatchExecutor(max_workers=128, batch_size=32)

            tasks = [
                Task(input_data=data, context={"id": i})
                for i, data in enumerate(dataset)
            ]

            def process(data):
                return call_api(data)

            def save_batch(results):
                for r in results:
                    if r.success:
                        db.save(r.result, r.context["id"])

            stats = executor.execute(tasks, process, save_batch)
        """
        # Initialize
        self.stats = Stats(total=len(tasks))
        original_handler = self._setup_interrupt_handler()
        results_buffer = []

        try:
            logger.info(
                f"Starting to process {self.stats.total} tasks "
                f"(max_workers={self.max_workers})"
            )

            # Use context manager for automatic cleanup
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks at once
                future_to_task = {
                    executor.submit(process_func, task.input_data): task
                    for task in tasks
                }

                # Main thread monitors and processes completed tasks
                iterator = as_completed(future_to_task)
                if self.show_progress_bar:
                    iterator = tqdm(
                        iterator, total=len(tasks), desc="Processing", smoothing=0.1
                    )

                # Process results as they complete (non-blocking)
                for future in iterator:
                    # Check for interrupt
                    if self.interrupted:
                        logger.info("Cancelling remaining tasks...")
                        self._cancel_remaining_tasks(future_to_task)
                        break

                    task = future_to_task[future]

                    # Process completed future
                    task_result = self._process_completed_future(
                        future, task, save_errors
                    )

                    if task_result is not None:
                        results_buffer.append(task_result)

                    self.stats.processed += 1

                    # Save batch if ready
                    self._save_batch_if_ready(results_buffer, save_func)

                    # Log progress periodically
                    if self.stats.processed % self.log_every_n_items == 0:
                        self.stats.log_progress()

        except KeyboardInterrupt:
            logger.warning("Detected KeyboardInterrupt")
            self.interrupted = True

        except Exception as e:
            logger.error(f"Execution error: {e}")
            raise

        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

            # Save remaining results
            self._save_remaining_results(results_buffer, save_func)

            # Log summary
            self.stats.log_summary()

        return self.stats


def run_batch_concurrent(
    tasks: list[Task[T]],
    process_func: Callable[[T], R],
    save_func: Callable[[list[TaskResult[R]]], None],
    max_workers: int = 128,
    batch_size: int = 32,
    show_progress_bar: bool = True,
    save_errors: bool = False,
) -> Stats:
    """
    Simplified API for concurrent batch execution.

    This is the main entry point that hides BatchExecutor details.

    Args:
        tasks: List of Task objects
        process_func: Function signature: (input_data: T) -> R
        save_func: Function signature: (results: list[TaskResult[R]]) -> None
        max_workers: Maximum concurrent threads (default: 128)
        batch_size: Batch save size (default: 32)
        show_progress_bar: Show tqdm progress bar (default: True)
        save_errors: Save failed task results (default: False)

    Returns:
        Execution statistics

    Example:
        tasks = [Task(input_data=img, context={"id": i}) for i, img in enumerate(images)]

        def annotate(img):
            return call_vllm(img, prompt)

        def save_batch(results):
            annotations = [
                create_annotation(r.result, r.context["id"])
                for r in results if r.success
            ]
            storager.save(annotations)

        stats = run_batch_concurrent(tasks, annotate, save_batch, max_workers=128)
        print(f"Success rate: {stats.success / stats.total * 100:.1f}%")
    """
    executor = BatchExecutor(
        max_workers=max_workers,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    return executor.execute(
        tasks=tasks,
        process_func=process_func,
        save_func=save_func,
        save_errors=save_errors,
    )

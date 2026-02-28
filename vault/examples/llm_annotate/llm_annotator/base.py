"""
Base classes for annotation tools and vault processing.

This module provides stable base classes for:
- AnnotateTool: Define custom annotation logic
- VaultAnnotator: High-performance batch processing with thread pool

API Stability: Base class interfaces are stable.
Extend these classes to create new annotation tools.
"""

import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from vault.backend.duckdb import DuckDBHandler
from vault.backend.lance import LanceTaker
from vault.schema import ID
from vault.schema.multimodal import Creator, SampleAnnotation
from vault.storage.lanceduck.multimodal import MultiModalStorager

from .utils import execute_with_retry


class AnnotateTool:
    """
    Base class for annotation tools.

    Subclass this to create custom annotation logic.
    Define prompt templates, VLM calls, and data preparation.

    Attributes:
        basic_name: Tool identifier (stable across versions)
        sample_type: 'image', 'text', or 'sequence'
        creator_meta: Metadata about annotation creator
        model_name: VLM model identifier
    """

    basic_name: str
    sample_type: str = "sequence"
    creator_meta: dict

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.name = f"{self.basic_name}_{self.model_name}"

        assert self.sample_type in ["image", "text", "sequence"]

    @property
    def creator(self):
        """Get Creator object with tool metadata."""
        return Creator.create(
            name=self.name,
            meta=dict(
                model_name=self.model_name,
                **self.creator_meta,
            ),
        )

    def _get_all_samples(self, storager: MultiModalStorager) -> List[Dict]:
        """
        Query all samples from vault based on sample_type.

        Args:
            storager: Vault storage interface

        Returns:
            List of sample dictionaries with IDs
        """
        if self.sample_type == "sequence":
            samples = storager.meta_handler.query_batch(
                "SELECT id as sample_id, id as sequence_id FROM sequences;"
            )
        elif self.sample_type == "image":
            samples = storager.meta_handler.query_batch(
                "SELECT image_id as sample_id, sequence_id FROM sequence_images;"
            )
        else:
            samples = storager.meta_handler.query_batch(
                "SELECT text_id as text_id, sequence_id FROM sequence_texts;"
            )

        samples = [{k: ID.from_(v) for k, v in s.items()} for s in samples]
        samples.sort(key=lambda x: x["sample_id"].to_int())
        return samples

    def _get_processed_ids(self, duckdb_handler: DuckDBHandler) -> set:
        """
        Query already processed sample IDs.

        Args:
            duckdb_handler: DuckDB interface for querying

        Returns:
            Set of processed sample IDs
        """
        name = self.name

        if self.sample_type == "sequence":
            sql = "select sequence_id as id from sample_annotations where name = ?;"
        else:
            sql = f"""
            SELECT
                sae.element_id AS id
            FROM
                sample_annotation_elements AS sae
            JOIN
                sample_annotations AS sa ON sae.sample_annotation_id = sa.id
            WHERE
                sa.name = ? AND sae.element_type = '{self.sample_type}';
            """

        processed_ids = {
            ID.from_(val["id"]) for val in duckdb_handler.query_batch(sql, (name,))
        }

        return processed_ids

    def _prepare_kwargs_and_participants(
        self, sample: Dict, storager: MultiModalStorager
    ):
        """
        Prepare API call arguments and annotation participants.

        Override this method to customize data preparation.

        Args:
            sample: Sample metadata dict
            storager: Vault storage interface

        Returns:
            Tuple of (participants, kwargs)
            - participants: Tuple of (id, type, role) for each element
            - kwargs: Dict of arguments for __call__
        """
        raise NotImplementedError()

    def prepare_single_sample(
        self, sample: Dict, storager: MultiModalStorager
    ) -> Dict | None:
        """
        Prepare a single sample for processing.

        Args:
            sample: Sample metadata
            storager: Vault storage interface

        Returns:
            Dict with sequence_id, participants, kwargs
        """
        participants, kwargs = self._prepare_kwargs_and_participants(sample, storager)

        if participants is None and kwargs is None:
            return None

        return {
            "sequence_id": ID.from_(sample["sequence_id"]),
            "participants": participants,
            "kwargs": kwargs,
        }

    def __call__(self, *args, **kwargs):
        """
        Execute annotation logic.

        Override this method to implement your VLM calls.

        Returns:
            JSON string of annotation result
        """
        raise NotImplementedError()


@dataclass
class VaultAnnotator:
    """
    High-performance batch annotation processor.

    Uses multi-threaded processing with producer-consumer pattern.
    Handles data loading, API calls, result batching, and atomic commits.

    Attributes:
        tool: AnnotateTool instance
        vault_path: Path to vault directory
        output_path: Path to output DuckDB file
        batch_size: Number of results to batch before saving
        max_workers: Thread pool size
        retry_count: API retry attempts
    """

    tool: AnnotateTool
    vault_path: Path | str
    output_path: Path | str
    batch_size: int
    max_workers: int = 32
    retry_count: int = 3

    def __post_init__(self):
        self.vault_path = Path(self.vault_path)
        self.output_path = Path(self.output_path)
        self.storager = MultiModalStorager(self.vault_path.as_posix(), read_only=True)

        # Queue attributes
        self.results_queue = queue.Queue()
        self.save_batch_size = self.batch_size
        self.stop_event = threading.Event()
        self.processed_count = 0
        self.total_to_process = 0
        self.start_time = None
        self.processing_executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Initialize storage
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lance_index()
        self.duckdb_handler = self._init_duckdb()

    def _init_lance_index(self):
        """Create BTREE index on Lance dataset for fast lookups."""
        lance_taker = LanceTaker()
        lance_dataset = lance_taker.lance_dataset(self.storager.lance_uris["images"])
        if lance_taker.exist_id_index(lance_dataset):
            lance_dataset.create_scalar_index(
                column="id", index_type="BTREE", name="id_btree_idx", replace=True
            )

    def _init_duckdb(self):
        """Initialize output DuckDB handler."""
        handler = DuckDBHandler(
            schema=self.storager.DUCKDB_SCHEMA,
            read_only=False,
            db_path=self.output_path,
        )
        handler.create()
        return handler

    def _get_processed_ids(self) -> set:
        """Get already processed sample IDs from output database."""
        if not Path(self.output_path).exists():
            return set()
        try:
            processed_ids = self.tool._get_processed_ids(self.duckdb_handler)

            logger.info(
                f"Found {len(processed_ids)} already processed items in {self.output_path}"
            )
            return processed_ids
        except Exception as e:
            logger.warning(
                f"Could not read existing DuckDB dataset at {self.output_path}. Starting from scratch. Error: {e}"
            )
            return set()

    @staticmethod
    def _generate_batches(items: List[Any], batch_size: int):
        """Generator that yields batches from list."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _save_batch_results(self, results: List[Dict]):
        """Save batch of results to DuckDB."""
        if not results:
            return

        sample_annotations = []

        for result in results:
            sample_annotation = SampleAnnotation.create(
                name=self.tool.name,
                creator=self.tool.creator,
                value=result["api_result"],
                sequence_id=result["sequence_id"],
                participants=result["participants"],
            )
            sample_annotations.append(sample_annotation)

        self.storager.add_sample_annotations(
            sample_annotations, duckdb_handler=self.duckdb_handler
        )
        logger.info(
            f"Successfully added {len(sample_annotations)} sample annotations to {self.output_path}"
        )

    def _data_preparation_and_processing(self, samples_to_process: List[Dict]):
        """Producer thread: prepare data and submit for processing."""
        try:
            for sample in samples_to_process:
                if self.stop_event.is_set():
                    break
                try:
                    prepared_sample = self.tool.prepare_single_sample(
                        sample, self.storager
                    )
                    if prepared_sample is not None:
                        # Submit to thread pool
                        self._submit_single_sample_for_processing(prepared_sample)

                except Exception as e:
                    logger.error(
                        f"准备数据时出错 {sample.get('sequence_id', 'unknown')}: {e}"
                    )
                    logger.debug(traceback.format_exc())
                    continue

        except Exception as e:
            logger.error(f"数据准备生产者线程出错: {e}")
            logger.debug(traceback.format_exc())
        finally:
            # Wait for all processing to complete
            self._wait_for_all_processing_complete()

    def _submit_single_sample_for_processing(self, prepared_sample: Dict):
        """
        Submit single sample to thread pool.

        Uses method reference instead of closure to avoid memory leaks.
        """
        try:
            # Extract necessary data
            sequence_id = prepared_sample["sequence_id"]
            participants = prepared_sample["participants"]
            kwargs = prepared_sample["kwargs"]

            # Key: Submit method reference with parameters, not closure
            # This allows kwargs to be garbage collected after task completion
            future = self.processing_executor.submit(
                self._process_single_task,
                sequence_id,
                participants,
                kwargs,
            )

            # Set callback for result handling
            def handle_result(future):
                try:
                    result = future.result()
                    if result is not None:
                        self.results_queue.put(result)
                    self.processed_count += 1

                    # Log progress
                    if self.processed_count % 10 == 0:
                        self._log_progress()

                except Exception as e:
                    logger.error(f"处理结果时出错: {e}")
                    self.processed_count += 1

            future.add_done_callback(handle_result)

        except Exception as e:
            logger.error(f"提交处理任务时出错: {e}")

    def _wait_for_all_processing_complete(self):
        """Wait for all processing tasks to complete."""
        pass

    def _process_single_task(self, sequence_id, participants, kwargs):
        """
        Process single task in worker thread.

        Note: Submitted as method reference to avoid closure memory leaks.
        After completion, kwargs is immediately garbage collected.
        """
        try:
            result = execute_with_retry(
                self.tool,
                retry_count=self.retry_count,
                retry_delay=2.0,
                **kwargs,
            )

            return {
                "sequence_id": sequence_id,
                "participants": participants,
                "api_result": result,
            }
        except Exception as e:
            logger.error(f"处理单个项目时出错: {e}")
            return None

    def _log_progress(self):
        """Log processing progress with speed and ETA."""
        if self.processed_count == 0 or self.start_time is None:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Calculate speed (items/sec)
        speed = self.processed_count / elapsed_time if elapsed_time > 0 else 0

        # Calculate progress percentage
        progress_percent = (
            (self.processed_count / self.total_to_process * 100)
            if self.total_to_process > 0
            else 0
        )

        # Calculate ETA
        remaining_items = self.total_to_process - self.processed_count
        eta_seconds = remaining_items / speed if speed > 0 else 0

        # Format time display
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}秒"
            elif seconds < 3600:
                return f"{seconds / 60:.1f}分钟"
            else:
                return f"{seconds / 3600:.1f}小时"

        eta_str = format_time(eta_seconds)
        elapsed_str = format_time(elapsed_time)

        logger.info(
            f"进度: {self.processed_count}/{self.total_to_process} "
            f"({progress_percent:.1f}%) | "
            f"速度: {speed:.2f}项目/秒 | "
            f"已用时: {elapsed_str} | "
            f"预计剩余: {eta_str}"
        )

    def _result_saver_consumer(self):
        """Consumer thread: save results in batches."""
        results_buffer = []

        try:
            while not self.stop_event.is_set():
                try:
                    # Get result from queue with timeout
                    result = self.results_queue.get(timeout=30)
                    if result is None:  # Termination signal
                        break

                    results_buffer.append(result)

                    # Save when buffer reaches batch size
                    if len(results_buffer) >= self.save_batch_size:
                        self._save_batch_results(results_buffer)
                        results_buffer = []

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"结果保存消费者线程出错: {e}")

        finally:
            # Save remaining results
            if results_buffer:
                self._save_batch_results(results_buffer)

    def run(self):
        """Execute annotation workflow with streaming processing."""
        logger.info(f"--- Starting processing for vault: {self.vault_path} ---")
        processed_ids = self._get_processed_ids()
        all_samples = self.tool._get_all_samples(self.storager)
        assert all("sequence_id" in s for s in all_samples)
        samples_to_process = [
            m for m in all_samples if m["sample_id"] not in processed_ids
        ]

        if not samples_to_process:
            logger.info("All items have already been processed. Nothing to do.")
            return

        self.total_to_process = len(samples_to_process)
        self.start_time = time.time()
        logger.info(
            f"Total items: {len(all_samples)}, To process: {self.total_to_process}"
        )

        # Start result saver consumer thread
        saver_thread = threading.Thread(target=self._result_saver_consumer, daemon=True)
        saver_thread.start()

        # Start data preparation and processing producer thread
        producer_thread = threading.Thread(
            target=self._data_preparation_and_processing,
            args=(samples_to_process,),
            daemon=True,
        )
        producer_thread.start()

        try:
            # Main thread waits for completion
            while (
                self.processed_count < self.total_to_process
                and not self.stop_event.is_set()
            ):
                time.sleep(1)

                # Log progress periodically
                if self.processed_count % 50 == 0 and self.processed_count > 0:
                    self._log_progress()

        except KeyboardInterrupt:
            logger.info("收到中断信号, 正在停止...")
            self.stop_event.set()
        finally:
            # Wait for producer thread to finish
            producer_thread.join(timeout=60)

            # Send termination signal to results queue
            self.results_queue.put(None)

            # Wait for saver thread to finish
            saver_thread.join(timeout=120)

            # Shutdown processing thread pool
            self.processing_executor.shutdown(wait=True)

            logger.info(f"--- Finished processing for vault: {self.vault_path} ---")

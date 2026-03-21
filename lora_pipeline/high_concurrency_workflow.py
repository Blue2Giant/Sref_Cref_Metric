import asyncio
import fnmatch
import json
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import aiohttp

from comfykit import ComfyKit
from comfykit.logger import logger


class QueueFullException(Exception):
    pass


@dataclass
class DownloadTask:
    task_id: str
    url: str
    save_dir: str
    expected_prefix: str
    created_at: float


@dataclass
class DownloadResult:
    task_id: str
    success: bool
    file_path: str
    error: str
    duration_ms: float


def _extract_filename(image_url: str) -> str:
    parsed = urlparse(image_url)
    query = parse_qs(parsed.query)
    return (query.get("filename") or [""])[0]


def _detect_image_type(data: bytes) -> str:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if len(data) >= 2 and data[:2] == b"BM":
        return "bmp"
    return ""


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


class BaseHighConcurrencyWorkflowProcessor(ABC):
    def __init__(
        self,
        endpoints: List[str],
        comfykit_session_pool_size: int = 1,
        acquire_blocking: bool = True,
        acquire_retries: int = 3,
        acquire_retry_interval_sec: float = 0.5,
        downloader_pool_size: int = 2,
        download_queue_size: int = 256,
        download_timeout_sec: int = 120,
    ):
        if not endpoints:
            raise ValueError("endpoints cannot be empty")
        self.endpoints = endpoints
        self.comfykit_session_pool_size = max(1, int(comfykit_session_pool_size))
        self.acquire_blocking = acquire_blocking
        self.acquire_retries = max(1, int(acquire_retries))
        self.acquire_retry_interval_sec = max(0.0, float(acquire_retry_interval_sec))
        self.downloader_pool_size = max(1, int(downloader_pool_size))
        self.download_queue_size = max(1, int(download_queue_size))
        self.download_timeout_sec = max(1, int(download_timeout_sec))
        self._temp_files: set[str] = set()
        self._comfykit_queue: asyncio.Queue[ComfyKit] = asyncio.Queue(maxsize=len(self.endpoints))
        self._comfykit_clients: List[ComfyKit] = []
        self._download_queue: asyncio.Queue[Optional[DownloadTask]] = asyncio.Queue(maxsize=self.download_queue_size)
        self._download_workers: List[asyncio.Task] = []
        self._download_results: Dict[str, DownloadResult] = {}
        self._started = False

    async def start(self):
        if self._started:
            return
        for endpoint in self.endpoints:
            kit = ComfyKit(comfyui_url=endpoint, session_pool_size=self.comfykit_session_pool_size)
            self._comfykit_clients.append(kit)
            self._comfykit_queue.put_nowait(kit)
        for worker_id in range(self.downloader_pool_size):
            self._download_workers.append(asyncio.create_task(self._download_worker(worker_id)))
        self._started = True

    async def stop(self):
        if not self._started:
            self.cleanup_temp_files()
            return
        for _ in range(len(self._download_workers)):
            await self._download_queue.put(None)
        if self._download_workers:
            await asyncio.gather(*self._download_workers, return_exceptions=True)
        for client in self._comfykit_clients:
            try:
                await client.close()
            except Exception as exc:
                logger.error(f"close ComfyKit failed: {exc}")
        self._download_workers.clear()
        self._comfykit_clients.clear()
        self._started = False
        self.cleanup_temp_files()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    def inject_workflow_params(
        self,
        workflow_json_path: str,
        target_node_id: str,
        prompt: str,
        positive: str,
        negative: str,
        random_seed: int,
        prefix: str,
        optional_fields: Optional[Dict[str, Any]] = None,
    ) -> str:
        with open(workflow_json_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        node_id = str(target_node_id)
        node = workflow.setdefault(node_id, {})
        inputs = node.setdefault("inputs", {})
        merged_fields: Dict[str, Any] = {}
        if prompt is not None:
            merged_fields["prompt"] = prompt
        if positive is not None:
            merged_fields["positive"] = positive
        if negative is not None:
            merged_fields["negative"] = negative
        if random_seed is not None:
            merged_fields["random_seed"] = random_seed
        if prefix is not None:
            merged_fields["prefix"] = prefix
        if optional_fields:
            merged_fields.update(optional_fields)
        _deep_merge(inputs, merged_fields)
        ts = int(time.time() * 1000)
        tmp_name = f"workflow_{ts}_{uuid.uuid4().hex}.json"
        tmp_path = os.path.abspath(os.path.join(gettempdir(), tmp_name))
        self._temp_files.add(tmp_path)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(workflow, f, ensure_ascii=False, indent=2)
        except Exception:
            self._temp_files.discard(tmp_path)
            raise
        return tmp_path

    def cleanup_temp_files(self):
        for file_path in list(self._temp_files):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as exc:
                logger.error(f"cleanup temp workflow failed: {file_path}, err={exc}")
            finally:
                self._temp_files.discard(file_path)

    @abstractmethod
    def build_custom_workflow(
        self,
        workflow_json_path: str,
        prompt: str,
        positive: str,
        negative: str,
        random_seed: int,
        prefix: str,
        optional_fields: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    async def _acquire_comfykit(self, blocking: Optional[bool] = None) -> ComfyKit:
        use_blocking = self.acquire_blocking if blocking is None else blocking
        last_error: Optional[Exception] = None
        for attempt in range(1, self.acquire_retries + 1):
            try:
                if use_blocking:
                    return await asyncio.wait_for(self._comfykit_queue.get(), timeout=0.5)
                return self._comfykit_queue.get_nowait()
            except asyncio.QueueEmpty as exc:
                last_error = exc
            except asyncio.TimeoutError as exc:
                last_error = exc
            if attempt < self.acquire_retries:
                await asyncio.sleep(self.acquire_retry_interval_sec)
        raise RuntimeError(f"acquire comfykit failed after {self.acquire_retries} retries: {last_error}")

    async def _release_comfykit(self, client: ComfyKit):
        await self._comfykit_queue.put(client)

    def _prefix_match(self, filename: str, expected_prefix: str) -> bool:
        if not expected_prefix:
            return True
        if expected_prefix.startswith("re:"):
            return re.search(expected_prefix[3:], filename) is not None
        if any(ch in expected_prefix for ch in ["*", "?", "["]):
            return fnmatch.fnmatch(filename, expected_prefix)
        return filename.startswith(expected_prefix)

    async def submit_download_task(self, url: str, save_dir: str, expected_prefix: str) -> bool:
        filename = _extract_filename(url)
        if not self._prefix_match(filename, expected_prefix):
            return False
        await self._enqueue_download_task(url, save_dir, expected_prefix)
        return True

    async def _enqueue_download_task(self, url: str, save_dir: str, expected_prefix: str) -> str:
        task_id = uuid.uuid4().hex
        task = DownloadTask(
            task_id=task_id,
            url=url,
            save_dir=save_dir,
            expected_prefix=expected_prefix,
            created_at=time.time(),
        )
        try:
            await asyncio.wait_for(self._download_queue.put(task), timeout=2.0)
        except asyncio.TimeoutError as exc:
            raise QueueFullException("download queue is full for more than 2 seconds") from exc
        return task_id

    async def _download_worker(self, worker_id: int):
        timeout = aiohttp.ClientTimeout(total=self.download_timeout_sec)
        connector = aiohttp.TCPConnector(limit=0, enable_cleanup_closed=True)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            while True:
                task = await self._download_queue.get()
                if task is None:
                    self._download_queue.task_done()
                    break
                start = time.perf_counter()
                try:
                    Path(task.save_dir).mkdir(parents=True, exist_ok=True)
                    async with session.get(task.url) as response:
                        if response.status != 200:
                            raise Exception(f"download failed: HTTP {response.status}")
                        data = await response.read()
                    if not data:
                        raise Exception("downloaded image is empty")
                    image_type = _detect_image_type(data)
                    if not image_type:
                        raise Exception("downloaded image is invalid")
                    save_path = os.path.abspath(os.path.join(task.save_dir, f"{task.task_id}.{image_type}"))
                    with open(save_path, "wb") as f:
                        f.write(data)
                    duration = round((time.perf_counter() - start) * 1000, 2)
                    self._download_results[task.task_id] = DownloadResult(
                        task_id=task.task_id,
                        success=True,
                        file_path=save_path,
                        error="",
                        duration_ms=duration,
                    )
                except Exception as exc:
                    duration = round((time.perf_counter() - start) * 1000, 2)
                    self._download_results[task.task_id] = DownloadResult(
                        task_id=task.task_id,
                        success=False,
                        file_path="",
                        error=str(exc),
                        duration_ms=duration,
                    )
                    logger.error(f"download worker {worker_id} failed: task={task.task_id}, err={exc}")
                finally:
                    self._download_queue.task_done()

    async def wait_download_tasks(self, task_ids: List[str], timeout_sec: int = 120) -> Dict[str, DownloadResult]:
        target = set(task_ids)
        start = time.perf_counter()
        while True:
            done = {task_id: self._download_results[task_id] for task_id in target if task_id in self._download_results}
            if len(done) == len(target):
                return done
            if (time.perf_counter() - start) > timeout_sec:
                raise TimeoutError(f"wait download tasks timeout, done={len(done)}, total={len(target)}")
            await asyncio.sleep(0.1)

    async def execute(
        self,
        workflow_json_path: str,
        prompt: str,
        positive: str,
        negative: str,
        random_seed: int,
        prefix: str,
        save_dir: str,
        optional_fields: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if not self._started:
            raise RuntimeError("processor is not started")
        temp_workflow = ""
        client: Optional[ComfyKit] = None
        download_task_ids: List[str] = []
        try:
            temp_workflow = self.build_custom_workflow(
                workflow_json_path=workflow_json_path,
                prompt=prompt,
                positive=positive,
                negative=negative,
                random_seed=random_seed,
                prefix=prefix,
                optional_fields=optional_fields or {},
            )
            client = await self._acquire_comfykit()
            result = await client.execute(temp_workflow)
            if result.status != "completed":
                raise RuntimeError(result.msg or f"workflow status={result.status}")
            for image_url in result.images or []:
                task_id = await self._enqueue_download_task(image_url, save_dir, prefix)
                download_task_ids.append(task_id)
            return download_task_ids
        finally:
            if client is not None:
                await self._release_comfykit(client)
            self.cleanup_temp_files()

    def __del__(self):
        try:
            self.cleanup_temp_files()
        except Exception:
            pass


class IllustriousSimpleWorkflowProcessor(BaseHighConcurrencyWorkflowProcessor):
    def build_custom_workflow(
        self,
        workflow_json_path: str,
        prompt: str,
        positive: str,
        negative: str,
        random_seed: int,
        prefix: str,
        optional_fields: Optional[Dict[str, Any]] = None,
    ) -> str:
        extra = dict(optional_fields or {})
        extra["filename_prefix"] = prefix
        temp_path = super().inject_workflow_params(
            workflow_json_path=workflow_json_path,
            target_node_id="25",
            prompt=prompt,
            positive=positive,
            negative=negative,
            random_seed=random_seed,
            prefix=prefix,
            optional_fields=extra,
        )
        with open(temp_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        if positive:
            workflow["18"]["inputs"]["text"] = positive
        elif prompt:
            workflow["18"]["inputs"]["text"] = prompt
        if negative:
            workflow["19"]["inputs"]["text"] = negative
        if random_seed is not None:
            workflow["5"]["inputs"]["seed"] = int(random_seed)
        if "loraname" in extra and extra["loraname"]:
            workflow["15"]["inputs"]["lora_name"] = extra["loraname"]
        workflow["25"]["inputs"]["filename_prefix"] = prefix
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(workflow, f, ensure_ascii=False, indent=2)
        return temp_path

import gc
import imghdr
import multiprocessing
import os
import random
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import megfile
import webdataset
from loguru import logger
from tqdm import tqdm

from vault.schema import ID
from vault.storage.lanceduck.multimodal import MultiModalStorager
from vault.utils import batched


def get_extension_imghdr(image_bytes: bytes) -> str:
    fmt = imghdr.what(None, h=image_bytes)
    ext_map = {
        "jpeg": "jpg",
        "png": "png",
        "gif": "gif",
        "bmp": "bmp",
        "tiff": "tiff",
        "webp": "webp",
    }
    if fmt is None:
        return "jpg"
    return ext_map.get(fmt, "jpg")


class ItemProvider:
    sources: list[str] | str

    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.storager = MultiModalStorager(vault_path, read_only=True)

        self.items = self.prepare_items()

    def prepare_items(self) -> list[dict]:
        sequence_ids = self.storager.get_sequence_ids_by_sources(self.sources)
        items = self.storager.get_sequence_metas(sequence_ids)
        return items

    def get_items(self, indices: list[int]) -> list[dict]:
        return [self.items[i] for i in indices]

    def get_images(self, ids: list[ID]) -> dict[ID, bytes]:
        return self.storager.get_image_bytes_by_ids(ids)


class MultiProcessWebDatasetWriter:
    """
    一个使用 ProcessPoolExecutor 和 webdataset.TarWriter 的多进程TAR包写入器。
    """

    def __init__(
        self,
        provider: ItemProvider,
        output_dir: str,
        samples_per_tar: int = 1000,
        num_processes: int | None = None,
    ):
        self.provider = provider
        self.output_dir = output_dir
        self.tmp_dir = "/tmp/export_as_tar/"
        self.samples_per_tar = samples_per_tar

        self.num_processes = num_processes or (os.cpu_count() or 1)

        megfile.smart_makedirs(self.output_dir, exist_ok=True)
        megfile.smart_makedirs(self.tmp_dir, exist_ok=True)
        logger.info(f"Output directory '{self.output_dir}' is ready.")
        logger.info(f"Using a process pool with max {self.num_processes} workers.")

    def _worker(self, indices_chunk: List[int], process_id: int):
        """
        每个进程执行的工作函数。
        现在使用 webdataset.TarWriter。
        """
        logger.info(
            f"[Worker {process_id}] Started. Handling {len(indices_chunk)} indices from {indices_chunk[0]} to {indices_chunk[-1]}."
        )
        start_time = time.time()

        for i in range(0, len(indices_chunk), self.samples_per_tar):
            sub_chunk = indices_chunk[i : i + self.samples_per_tar]
            if not sub_chunk:
                continue

            items = self.provider.get_items(sub_chunk)

            tar_filename = f"{uuid.uuid4()}.tar"
            cache_tar_filepath = megfile.smart_path_join(self.tmp_dir, tar_filename)
            tar_filepath = megfile.smart_path_join(self.output_dir, tar_filename)

            try:
                batch_size = 8
                with webdataset.TarWriter(cache_tar_filepath) as tar:
                    for batch_items in tqdm(
                        batched(items, batch_size=batch_size),
                        total=len(items) // batch_size,
                        position=process_id % 8 + 1,
                        leave=True,
                        desc=f"{process_id=} {tar_filename=}",
                    ):
                        image_ids = [item["image_id"] for item in batch_items]
                        id_to_image_bytes = self.provider.get_images(image_ids)
                        for item in batch_items:
                            assert "__key__" in item, "item must have __key__"

                            image_id = item.pop("image_id")
                            if image_id not in id_to_image_bytes:
                                logger.error(
                                    f"[Worker {process_id}] Image {image_id} not found."
                                )
                                continue
                            image_bytes = id_to_image_bytes[image_id]

                            tar.write(
                                {
                                    "__key__": item["__key__"],
                                    "json": item["json"],
                                    get_extension_imghdr(image_bytes): image_bytes,
                                }
                            )
                        del id_to_image_bytes

                megfile.smart_move(cache_tar_filepath, tar_filepath)

                logger.info(
                    f"[Worker {process_id}] Successfully created {tar_filepath} with {len(sub_chunk)} samples."
                )

            except Exception as e:
                import traceback

                logger.error(traceback.format_exc())
                logger.error(
                    f"[Worker {process_id}] Failed to create TAR file {tar_filepath}: {e}"
                )

            gc.collect()

        end_time = time.time()
        logger.info(
            f"[Worker {process_id}] Finished in {end_time - start_time:.2f} seconds."
        )
        return f"Worker {process_id} completed successfully."

    def run(self, total_indices: List[int]):
        """
        启动多进程任务。
        现在使用 ProcessPoolExecutor。
        """
        if not total_indices:
            logger.warning("Warning: No indices provided to process.")
            return

        logger.info(f"Total indices to process: {len(total_indices)}")

        # 均匀切分索引
        n = len(total_indices)
        k = self.num_processes
        random.shuffle(total_indices)
        chunks = [total_indices[n * i // k : n * (i + 1) // k] for i in range(k)]
        chunks = [c for c in chunks if c]

        # 使用 ProcessPoolExecutor 管理进程
        with ProcessPoolExecutor(
            max_workers=self.num_processes,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            logger.info(f"Submitting {len(chunks)} tasks to the process pool...")
            # 提交任务
            futures = {
                executor.submit(self._worker, chunk, i): i
                for i, chunk in enumerate(chunks)
            }

            # 等待任务完成并处理结果/异常
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    result = future.result()
                    logger.info(result)
                except Exception as exc:
                    logger.error(f"Worker {worker_id} generated an exception: {exc}")

        logger.info("\nAll tasks have been processed.")


class RewardProvider(ItemProvider):
    sources: list[str] | str = [
        "250910-rewards/HPSv3/HPSv2",
        "250910-rewards/第一批标注数据",
        "250910-rewards/HPSv3/real",
        "250910-rewards/250717_real_1",
        "250910-rewards/HPDv2/train",
        "250910-rewards/HPSv3",
        "250910-rewards/rapidata/t2i_human_preference",
    ]

    def prepare_items(self) -> list[dict]:
        handler = self.storager.meta_handler
        with handler:
            items = handler.query_batch(
                """
                SELECT DISTINCT
                t.content AS caption_cn,
                    si.image_id
                FROM sequences s
                JOIN sequence_texts st ON s.id = st.sequence_id
                JOIN texts t ON st.text_id = t.id
                JOIN sequence_images si ON s.id = si.sequence_id
                WHERE s.source IN ?
                AND st.index = 'caption_cn';
                """,
                [tuple(self.sources)],
            )
        items = [
            {
                "__key__": f"{item['image_id']}",
                "json": {"caption": item["caption_cn"]},
                "image_id": ID.from_(item["image_id"]),
            }
            for item in items
        ]
        return items


class tooopenProvider(ItemProvider):
    sources: list[str] | str = [
        "250820-tooopen_text",
    ]

    def prepare_items(self) -> list[dict]:
        handler = self.storager.meta_handler
        with handler:
            _items = handler.query_batch(
                """
                SELECT DISTINCT
                    si.image_id,
                    i.uri AS image_uri,
                    t_cn.content AS caption_cn,
                    t_en.content AS caption_en
                FROM sequences s
                JOIN sequence_images si ON s.id = si.sequence_id
                JOIN images i ON si.image_id = i.id
                -- 中文 caption
                JOIN sequence_texts st_cn ON s.id = st_cn.sequence_id
                JOIN texts t_cn ON st_cn.text_id = t_cn.id
                -- 英文 caption
                JOIN sequence_texts st_en ON s.id = st_en.sequence_id
                JOIN texts t_en ON st_en.text_id = t_en.id
                WHERE s.source IN ?
                AND st_cn.index = 'caption_cn'
                AND st_en.index = 'caption_en';
                """,
                [tuple(self.sources)],
            )
        items = []
        for item in _items:
            if "cn" in megfile.SmartPath(item["image_uri"]).stem:
                items.append(
                    dict(
                        __key__=f"{item['image_id']}",
                        json={"caption": item["caption_cn"]},
                        image_id=ID.from_(item["image_id"]),
                    )
                )
            else:
                items.append(
                    dict(
                        __key__=f"{item['image_id']}",
                        json={"caption": item["caption_en"]},
                        image_id=ID.from_(item["image_id"]),
                    )
                )
        return items


class textualimageProvider(ItemProvider):
    sources: list[str] | str = [
        # "250901-textual_image-xp/mix-text",
        # "250901-textual_image-xp/fluxdevtext",
        # "250901-textual_image-xp/mix-text-en",
        "250925-mj"
    ]

    def prepare_items(self) -> list[dict]:
        handler = self.storager.meta_handler
        with handler:
            items = handler.query_batch(
                """
                SELECT DISTINCT
                t.content AS caption,
                    si.image_id
                FROM sequences s
                JOIN sequence_texts st ON s.id = st.sequence_id
                JOIN texts t ON st.text_id = t.id
                JOIN sequence_images si ON s.id = si.sequence_id
                WHERE s.source IN ?
                AND st.index = 'caption';
                """,
                [tuple(self.sources)],
            )
        items = [
            {
                "__key__": f"{item['image_id']}",
                "json": {"caption": item["caption"]},
                "image_id": ID.from_(item["image_id"]),
            }
            for item in items
        ]
        return items


def main(
    output_dir: str = "s3://xp-base/datasets/unclean/distill-qwenimage/250925-mj",
):
    logger.info("\nStarting the multi-process WebDataset writing script...")

    provider = textualimageProvider("/mnt/jfs/datasets/vault/t2i/20250909-qwen-image/")

    logger.info(f"Total items: {len(provider.items)}")

    writer = MultiProcessWebDatasetWriter(provider, output_dir, 1000, 20)

    overall_start_time = time.time()
    writer.run(list(range(len(provider.items))))
    total_time = time.time() - overall_start_time

    logger.info(
        f"Script finished. Total execution time: {total_time:.2f} seconds. {len(provider.items) / total_time:.2f} items/s"
    )
    logger.info(f"Check the output in the '{output_dir}' directory.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

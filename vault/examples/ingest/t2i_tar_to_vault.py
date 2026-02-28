import abc
import io
import json
from dataclasses import dataclass
from typing import Any, Generator

import megfile
import PIL.Image
import webdataset as wds
from loguru import logger
from tqdm import tqdm

import vault
import vault.schema.multimodal as multimodal
from vault.schema import ID
from vault.storage.lanceduck.multimodal import MultiModalStorager
from vault.utils.ingest import expand_tar_urls, run_concurrently

awds = vault.awds


@dataclass
class BaseIngestor(abc.ABC):
    """
    数据入库流程的抽象基类。

    子类需要实现 `prepare_tasks` 和 `process_tasks` 方法。
    """

    name: str
    vault_path: str

    task_chunk_size: int
    num_workers: int

    @abc.abstractmethod
    def prepare_tasks(self) -> list[Any]:
        """
        准备所有需要处理的任务单元。

        例如: 返回一个包含所有 tar 文件路径的列表。

        Returns:
            list[Any]: 任务列表。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def process_tasks(
        self, tasks_chunk: list[Any], worker_id: int
    ) -> Generator[multimodal.PackSequence, None, None]:
        """
        处理一小批任务，并生成 PackSequence 对象。
        这个方法将在并发的子进程中被调用。

        Args:
            tasks_chunk (list[Any]): 由 `run` 方法分配的一批任务。
            worker_id (int): 当前工作进程的 ID, 可用于 TQDM 定位。

        Yields:
            multimodal.PackSequence: 处理完成的数据包。
        """
        raise NotImplementedError

    def _worker_ingest(self, tasks_chunk: list[Any], worker_id: int):
        """
        每个工作进程执行的内部函数。
        它会创建一个 Storager 实例，并调用 process_tasks 来处理数据。
        """
        storager = MultiModalStorager(self.vault_path)
        sequences = self.process_tasks(tasks_chunk, worker_id)
        storager.add_sequences(sequences)

    def run(self, num_workers: int | None = None, chunk_size: int | None = None):
        """
        执行数据入库的主流程。

        Args:
            num_workers (int): 并发工作进程的数量。
            chunk_size (int): 每个工作进程一次处理的任务数量。
        """
        logger.info(f"[{self.name}] 开始数据入库流程...")

        # 1. 初始化
        MultiModalStorager.init(self.vault_path)

        # 2. 准备任务
        all_tasks = self.prepare_tasks()
        if not all_tasks:
            logger.warning("`prepare_tasks` 未返回任何任务，流程结束。")
            return

        logger.info(f"[{self.name}] 准备了 {len(all_tasks)} 个任务单元。")

        chunk_size = chunk_size or self.task_chunk_size
        self.task_chunk_size = chunk_size

        # 3. 将任务分块
        task_chunks = [
            all_tasks[i : i + chunk_size] for i in range(0, len(all_tasks), chunk_size)
        ]
        logger.info(
            f"任务被分为 {len(task_chunks)} 个块，每块最多包含 {chunk_size} 个任务单元。"
        )

        num_workers = num_workers or self.num_workers
        self.num_workers = num_workers

        # 4. 执行
        run_concurrently(self._worker_ingest, task_chunks, num_workers)

        # 5. 收尾
        logger.info(f"[{self.name}] 所有并发任务已完成，开始提交数据...")
        MultiModalStorager(self.vault_path).commit()
        logger.info(f"[{self.name}] 数据提交成功！入库流程结束。")


@dataclass
class EchoGPT4o(BaseIngestor):
    name: str = "Echo-4o-Image"
    vault_path: str = "/mnt/jfs/datasets/vault/t2i/20250903-sft-ai"

    task_chunk_size: int = 1
    num_workers: int = 32

    root: str = "s3+b://collect-data-datasets/202508/huggingface/huggingface_data_downloader-ppKG5gzp/resources/batch_1/8AOwY9nb/"
    tar_folder: str = megfile.smart_path_join(root, "Yejy53/Echo-4o-Image/")

    folders: tuple = ("Surrel-Fantasy-Image", "Instruction-Following-Image")

    def prepare_tasks(self) -> list[Any]:
        tar_urls = expand_tar_urls(
            [
                megfile.smart_path_join(self.tar_folder, folder, "images")
                for folder in self.folders
            ]
        )
        return tar_urls

    def process_tasks(
        self, tasks_chunk: list[Any], worker_id: int
    ) -> Generator[multimodal.PackSequence, None, None]:
        jsonls = (
            megfile.smart_path_join(
                self.tar_folder, "Surrel-Fantasy-Image/conflict.jsonl"
            ),
            megfile.smart_path_join(
                self.tar_folder,
                "Instruction-Following-Image/Instruction-Following-Image.jsonl",
            ),
        )

        _annotations = dict()
        _captions = dict()

        for jsonl in jsonls:
            source = f"{self.name}/{[f for f in self.folders if f in jsonl][0]}"

            with megfile.smart_open(jsonl, "rt") as f:
                for line in f.readlines():
                    data = json.loads(line)

                    if data["type"] not in _annotations:
                        _annotations[data["type"]] = multimodal.Annotation.create(
                            name=data["type"],
                            meta=dict(source=self.name),
                            type_="prompt类型",
                        )

                    _captions[data["output_image"]] = multimodal.Text.create(
                        content=data["instruction"],
                        uri=data["output_image"],
                        source=source,
                        language="en",
                        annotations=[_annotations[data["type"]]],
                    )

        generated_by_gpt4o = multimodal.Annotation.generated_by(model="gpt4o")

        for tar_url in tasks_chunk:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(tar_url),
                awds.atarfile_to_samples(handler=awds.print_exception_and_continue),
                wds.rename(
                    image=awds.IMAGE_EXTENSIONS_LISTSTR,
                    handler=awds.print_exception_and_continue,
                ),
            )

            for sample in tqdm(
                dataset,
                desc=f"{tar_url.replace(self.root, 'hf://datasets/')}",
                position=worker_id % 8 + 1,
                leave=True,
            ):
                __url__ = sample["__url__"].replace(self.root, "hf://datasets/")
                source = f"{self.name}/{[f for f in self.folders if f in __url__][0]}"

                uri = f"/{source}/images/{sample['__key__']}.jpg"

                image = multimodal.Image.create(
                    sample["image"],
                    uri=uri,
                    source=source,
                    annotations=[generated_by_gpt4o],
                )
                caption = _captions[uri]

                yield multimodal.PackSequence.from_text_to_image(
                    caption=caption,
                    image=image,
                    source=source,
                    uri=uri,
                    meta=dict(tar_url=__url__, tar_key=sample["__key__"]),
                )

            logger.debug(f"{tar_url} done.")


@dataclass
class QwenTextualImage(BaseIngestor):
    name: str = "250820-tooopen_text"
    vault_path: str = "/mnt/jfs/datasets/vault/t2i/20250909-qwen-image"

    task_chunk_size: int = 500
    num_workers: int = 32

    root: str = "s3://xp-base/tests/t2i/assets/distill_qwenimage/Qwenimage_20250820_ASCEND_Qwen-image_infer-visu-teepon-text--cfg3.5--truecfg4.0-1328_1328_step50_each1_nosp--it0--250820-tooopen_text"
    jsonl_path: str = (
        "s3://xp-base/datasets/sft-20250321-iqaclean-prompts/250820-tooopen_text.jsonl"
    )

    @staticmethod
    def parse_jsonl(jsonl_path):
        import pandas

        def generate_image_uri(item):
            tar = item["tar_url"].split("/")[-1].split(".tar")[0]
            name = f"{tar}_{item['tar_key']}"
            return name

        with megfile.smart_open(jsonl_path, "rt") as f:
            df = pandas.read_json(f, lines=True)
            df["image_uri"] = df.apply(generate_image_uri, axis=1)

            df_clean = df[df["image_uri"].map(df["image_uri"].value_counts()) == 1]  # type: ignore

            return df_clean.set_index("image_uri")[
                ["caption", "tar_url", "tar_key", "source"]
            ].to_dict("index")  # type: ignore

    def prepare_tasks(self) -> list[Any]:
        image_files = megfile.smart_listdir(self.root)

        group_meta = self.parse_jsonl(self.jsonl_path)
        groups = dict()

        for image_file in image_files:
            image_uri = image_file.rsplit("_", maxsplit=1)[0]
            if image_uri in group_meta:
                if image_uri in groups:
                    groups[image_uri]["images"].append(
                        megfile.smart_path_join(self.root, image_file)
                    )
                else:
                    groups[image_uri] = dict(
                        images=[megfile.smart_path_join(self.root, image_file)],
                        uri=image_uri,
                        **group_meta[image_uri],
                    )

        group_tasks = list(groups.values())
        return group_tasks

    def process_tasks(
        self, tasks_chunk: list[Any], worker_id: int
    ) -> Generator[multimodal.PackSequence, None, None]:
        generated_by_qwen = multimodal.Annotation.generated_by(model="qwen-image")

        for group in tqdm(
            tasks_chunk, position=worker_id % 8 + 1, desc=f"proc={worker_id}"
        ):
            uri = group["uri"]
            images: list[multimodal.Image] = []
            source = f"{self.name}/{group['source']}"

            for f in group["images"]:
                images.append(
                    multimodal.Image.create(
                        megfile.smart_load_content(f),
                        uri=f,
                        annotations=[generated_by_qwen],
                        source=source,
                    )
                )

            caption = multimodal.Text.create(
                content=group["caption"],
                uri=uri,
                source=source,
            )

            yield multimodal.PackSequence.from_t2i_reward(
                caption=caption,
                image=images,
                source=source,
                uri=uri,
                meta=dict(tar_url=group["tar_url"], tar_key=group["tar_key"]),
            )


@dataclass
class QwenTextualImage2(BaseIngestor):
    name: str = "250910-rewards"
    vault_path: str = "/mnt/jfs/datasets/vault/t2i/20250909-qwen-image"

    task_chunk_size: int = 500
    num_workers: int = 16

    root: str = "s3://xp-base/tests/t2i/assets/distill_qwenimage/Qwenimage_20250820_ASCEND_Qwen-image_infer-visu-teepon-text--cfg3.5--truecfg4.0-1328_1328_step50_each1_nosp--it0--rewards_cn"
    jsonl_path: str = "s3://ruiwang/asserts/step-image/reject_sampling/rewards_cn.jsonl"

    jsonl_names: tuple = ("english_prompt", "prompt", "source")

    def parse_jsonl(self, jsonl_path):
        import pandas

        def generate_image_uri(item):
            name = f"{item['id']}"
            return name

        with megfile.smart_open(jsonl_path, "rt") as f:
            df = pandas.read_json(f, lines=True)
            df["image_uri"] = df.apply(generate_image_uri, axis=1)

            df_clean = df[df["image_uri"].map(df["image_uri"].value_counts()) == 1]  # type: ignore

            return df_clean.set_index("image_uri")[[*self.jsonl_names]].to_dict("index")  # type: ignore

    def prepare_tasks(self) -> list[Any]:
        image_files = megfile.smart_listdir(self.root)

        group_meta = self.parse_jsonl(self.jsonl_path)
        groups = dict()

        for image_file in image_files:
            image_uri = image_file.split("_", maxsplit=1)[0]
            if image_uri in group_meta:
                if image_uri in groups:
                    groups[image_uri]["images"].append(
                        megfile.smart_path_join(self.root, image_file)
                    )
                else:
                    groups[image_uri] = dict(
                        images=[megfile.smart_path_join(self.root, image_file)],
                        uri=image_uri,
                        **group_meta[image_uri],
                    )

        group_tasks = list(groups.values())
        return group_tasks

    def process_tasks(
        self, tasks_chunk: list[Any], worker_id: int
    ) -> Generator[multimodal.PackSequence, None, None]:
        generated_by_qwen = multimodal.Annotation.generated_by(model="qwen-image")

        for group in tqdm(
            tasks_chunk, position=worker_id % 8 + 1, desc=f"proc={worker_id}"
        ):
            uri = group["uri"]
            images: list[tuple[multimodal.Image, str]] = []
            source = f"{self.name}/{group['source']}"

            for f in group["images"]:
                images.append(
                    (
                        multimodal.Image.create(
                            megfile.smart_load_content(f),
                            uri=megfile.SmartPath(f).name,
                            annotations=[generated_by_qwen],
                            source=source,
                        ),
                        "image_cn",
                    )
                )
            texts = [
                (
                    multimodal.Text.create(
                        content=group["prompt"],
                        uri=uri,
                        source=source,
                        language="zh",
                    ),
                    "caption_cn",
                ),
                (
                    multimodal.Text.create(
                        content=group["english_prompt"],
                        uri=uri,
                        source=source,
                        language="en",
                    ),
                    "caption_en",
                ),
            ]

            yield multimodal.PackSequence.create(
                images=images,
                texts=texts,
                source=source,
                uri=uri,
            )


@dataclass
class QwenTextualImage3(BaseIngestor):
    name: str = "250915-sft-en"
    vault_path: str = "/mnt/jfs/datasets/vault/t2i/20250909-qwen-image"

    task_chunk_size: int = 500
    num_workers: int = 16

    root: str = "s3+b://xp-base/tests/t2i/assets/distill_qwenimage/Qwenimage_20250820_ASCEND_Qwen-image_infer-visu-teepon-text--cfg3.5--truecfg4.0-1328_1328_step50_each1_nosp_valen--it0--20250904--sft-20250321-iqaclean-prompts_human"
    jsonl_path: str = (
        "s3://xp-base/datasets/20250904--sft-20250321-iqaclean-prompts_human/"
    )

    def parse_jsonl(self, jsonl_path):
        items = {}

        for json_path in megfile.smart_glob(
            megfile.smart_path_join(jsonl_path, "**/*.json")
        ):
            print(json_path)
            with megfile.smart_open(json_path, "rt") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON file: {json_path}")
                    continue

                for k, prompt in data.items():
                    _id = k

                    if _id in items:
                        logger.warning(f"Duplicate id: {_id}")
                        items.pop(_id)
                        continue

                    items[_id] = prompt

        return items

    def prepare_tasks(self) -> list[Any]:
        image_files = megfile.smart_listdir(self.root)

        group_meta = self.parse_jsonl(self.jsonl_path)
        groups = dict()

        meta_handler = MultiModalStorager(self.vault_path).meta_handler
        with meta_handler:
            items = meta_handler.query_batch(
                "SELECT uri FROM images WHERE source = ?",
                [self.name],
            )
            seen_images = {item["uri"] for item in items}

        logger.info(f"Found {len(seen_images)} images in vault")

        for image_file in image_files:
            image_uri = image_file.rsplit("_", maxsplit=1)[0]
            if image_uri in group_meta:
                image_path = megfile.smart_path_join(self.root, image_file)
                if megfile.SmartPath(image_path).name in seen_images:
                    continue

                if image_uri in groups:
                    groups[image_uri]["images"].append(image_path)
                else:
                    groups[image_uri] = dict(
                        images=[image_path],
                        uri=image_uri,
                        prompt=group_meta[image_uri],
                    )

        group_tasks = list(groups.values())
        return group_tasks

    def process_tasks(
        self, tasks_chunk: list[Any], worker_id: int
    ) -> Generator[multimodal.PackSequence, None, None]:
        generated_by_qwen = multimodal.Annotation.generated_by(model="qwen-image")

        for group in tqdm(
            tasks_chunk, position=worker_id % 8 + 1, desc=f"proc={worker_id}"
        ):
            uri = group["uri"]
            images: list[tuple[multimodal.Image, str]] = []
            source = f"{self.name}"

            for f in group["images"]:
                images.append(
                    (
                        multimodal.Image.create(
                            megfile.smart_load_content(f),
                            uri=megfile.SmartPath(f).name,
                            annotations=[generated_by_qwen],
                            source=source,
                        ),
                        "image",
                    )
                )
            texts = [
                (
                    multimodal.Text.create(
                        content=group["prompt"],
                        uri=uri,
                        source=source,
                    ),
                    "caption",
                ),
            ]

            yield multimodal.PackSequence.create(
                images=images,
                texts=texts,
                source=source,
                uri=uri,
            )


@dataclass
class MJTextualImage(BaseIngestor):
    name: str = "250925-mj"
    vault_path: str = "/mnt/jfs/datasets/vault/t2i/20250909-qwen-image"

    task_chunk_size: int = 500
    num_workers: int = 16

    root: str = "s3://xp-base/tests/t2i/assets/MJ_prompt/Qwenimage_20250820_ASCEND_Qwen-image_infer-visu-teepon-text--cfg3.5--truecfg4.0-1328_1328_step50_each1_nosp_val2--it0--MJ_prompt"
    jsonl_path: str = "s3://xp-base/datasets/prompts/MJ_prompt"

    def parse_jsonl(self, jsonl_path):
        items = {}

        for json_path in megfile.smart_glob(
            megfile.smart_path_join(jsonl_path, "**/*.json")
        ):
            print(json_path)
            with megfile.smart_open(json_path, "rt") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON file: {json_path}")
                    continue

                for k, prompt in data.items():
                    _id = f"{megfile.SmartPath(json_path).stem}_{k}"

                    if _id in items:
                        logger.warning(f"Duplicate id: {_id}")
                        items.pop(_id)
                        continue

                    items[_id] = prompt

        return items

    def prepare_tasks(self) -> list[Any]:
        image_files = megfile.smart_listdir(self.root)

        group_meta = self.parse_jsonl(self.jsonl_path)
        groups = dict()

        meta_handler = MultiModalStorager(self.vault_path).meta_handler
        with meta_handler:
            items = meta_handler.query_batch(
                "SELECT uri FROM images WHERE source = ?",
                [self.name],
            )
            seen_images = {item["uri"] for item in items}

        logger.info(f"Found {len(seen_images)} images in vault")

        for image_file in image_files:
            image_uri = image_file.rsplit("_", maxsplit=1)[0]
            if image_uri in group_meta:
                image_path = megfile.smart_path_join(self.root, image_file)
                if megfile.SmartPath(image_path).name in seen_images:
                    continue

                if image_uri in groups:
                    groups[image_uri]["images"].append(image_path)
                else:
                    groups[image_uri] = dict(
                        images=[image_path],
                        uri=image_uri,
                        prompt=group_meta[image_uri],
                    )

        group_tasks = list(groups.values())
        return group_tasks

    def process_tasks(
        self, tasks_chunk: list[Any], worker_id: int
    ) -> Generator[multimodal.PackSequence, None, None]:
        generated_by_qwen = multimodal.Annotation.generated_by(model="qwen-image")

        for group in tqdm(
            tasks_chunk, position=worker_id % 8 + 1, desc=f"proc={worker_id}"
        ):
            uri = group["uri"]
            images: list[tuple[multimodal.Image, str]] = []
            source = f"{self.name}"

            for f in group["images"]:
                images.append(
                    (
                        multimodal.Image.create(
                            megfile.smart_load_content(f),
                            uri=megfile.SmartPath(f).name,
                            annotations=[generated_by_qwen],
                            source=source,
                        ),
                        "image",
                    )
                )
            texts = [
                (
                    multimodal.Text.create(
                        content=group["prompt"],
                        uri=uri,
                        source=source,
                    ),
                    "caption",
                ),
            ]

            yield multimodal.PackSequence.create(
                images=images,
                texts=texts,
                source=source,
                uri=uri,
            )


@dataclass
class CIRR:
    name: str = "250905-cirr"
    vault_path: str = "/mnt/jfs/datasets/vault/composed_image_retrieval/20250905-cirr"
    root: str = "s3://collect-data-datasets/202509/huggingface/huggingface_data_downloader-4Bohx0TV/resources/batch_1/-U5i8y0q/BUAADreamer/cir_dataset/cirr_dataset/"

    def _generate_sequences(self, split_name: str, json_path: str):
        source: str = f"{self.name}/{split_name}"

        with megfile.smart_open(
            megfile.smart_path_join(self.root, json_path), "rt"
        ) as f:
            data = json.load(f)

            for item in data:
                query = multimodal.Text.create(
                    content=item["caption"],
                    uri=f"{source}/{item['pairid']}",
                    source=source,
                    language="en",
                )

                yield multimodal.PackSequence.create(
                    images=[],
                    texts=[(query, "query")],
                    source=source,
                    uri=f"{source}/{item['pairid']}",
                    meta=item,
                )

    def _generate_images(self):
        dataset = wds.DataPipeline(
            wds.SimpleShardList(megfile.smart_path_join(self.root, "images.tar.gz")),
            awds.atarfile_to_samples(handler=awds.print_exception_and_continue),
            wds.rename(
                image=awds.IMAGE_EXTENSIONS_LISTSTR,
                handler=awds.print_exception_and_continue,
            ),
        )

        for sample in tqdm(dataset, desc="Generating images"):
            yield multimodal.Image.create(
                sample["image"],
                uri=sample["__key__"],
                source=self.name,
            )

    def run(self):
        MultiModalStorager.init(self.vault_path)

        storager = MultiModalStorager(self.vault_path)

        for split_name, json_path in [
            ("val", "cirr/captions/cap.rc2.val.json"),
            ("test", "cirr/captions/cap.rc2.test1.json"),
            ("train", "cirr/captions/cap.rc2.train.json"),
        ]:
            storager.add_sequences(self._generate_sequences(split_name, json_path))
            logger.info(f"Added {split_name} sequences")

        storager.add_images(self._generate_images())
        logger.info("Added images")
        storager.commit()

    def associate(self):
        # 其实完全没必要搞这个函数, 本身可以在run里面一次写入。
        # 这里只是为了演示先写入，再关联。
        storager = MultiModalStorager(self.vault_path)

        handler = storager.meta_handler

        result_df = handler.conn.execute(
            f"SELECT id, uri FROM images WHERE source = '{self.name}'"
        ).fetch_df()

        # uri -> image_id的映射字典
        uri_to_id_mapping = dict(zip(result_df["uri"], result_df["id"]))

        print(f"找到 {len(uri_to_id_mapping)} 张图片")
        print("URI -> ID 映射示例:")
        for uri, image_id in list(uri_to_id_mapping.items())[:5]:  # 显示前5个
            print(f"  {uri} -> {image_id}")

        name_to_image_id = {}

        for json_path in [
            "split.rc2.test1.json",
            "split.rc2.train.json",
            "split.rc2.val.json",
        ]:
            with megfile.smart_open(
                megfile.smart_path_join(self.root, "cirr/image_splits", json_path)
            ) as f:
                data = json.load(f)
                for name, _uri in data.items():
                    _uri: str
                    uri = _uri.removeprefix("./").removesuffix(".png")
                    name_to_image_id[name] = ID.from_uuid(uri_to_id_mapping[uri])

        associations = []
        for split_name, json_path in [
            ("val", "cirr/captions/cap.rc2.val.json"),
            ("test", "cirr/captions/cap.rc2.test1.json"),
            ("train", "cirr/captions/cap.rc2.train.json"),
        ]:
            source: str = f"{self.name}/{split_name}"

            result_df = handler.conn.execute(
                f"SELECT id, uri FROM sequences WHERE source = '{source}'"
            ).fetch_df()
            uri_to_id_mapping = dict(zip(result_df["uri"], result_df["id"]))

            with megfile.smart_open(
                megfile.smart_path_join(self.root, json_path), "rt"
            ) as f:
                data = json.load(f)
                for item in data:
                    uri = f"{source}/{item['pairid']}"
                    try:
                        seq_id = ID.from_uuid(uri_to_id_mapping[uri])
                    except KeyError:
                        logger.warning(f"Image {uri} not found")
                        breakpoint()
                        break

                    for image_name in item["img_set"]["members"]:
                        associations.append(
                            (
                                name_to_image_id[image_name],
                                multimodal.PackSequenceIndex(
                                    sequence_id=seq_id, index="gallery"
                                ),
                            )
                        )

        storager.associate_images(associations)
        logger.info("Added associations")

    def refined_CIRR(self):
        storager = MultiModalStorager(self.vault_path)

        handler = storager.meta_handler

        result_df = handler.conn.execute("SELECT id, uri FROM sequences").fetch_df()

        uri_to_id_mapping = dict(zip(result_df["uri"], result_df["id"]))

        texts = []

        with megfile.smart_open(
            "/mnt/jfs/datasets/chuonghm--Refined-CIRR/refined_cirr.val.json"
        ) as f:
            data = json.load(f)

            for item in data:
                if not item["is_refined"]:
                    continue

                sequence_id = None
                for source in [
                    f"{self.name}/{split_name}"
                    for split_name in ["val", "test", "train"]
                ]:
                    if f"{source}/{item['pairid']}" in uri_to_id_mapping:
                        sequence_id = ID.from_uuid(
                            uri_to_id_mapping[f"{source}/{item['pairid']}"]
                        )
                        break

                if sequence_id is None:
                    logger.warning(f"Sequence {item['pairid']} not found")
                    continue

                texts.append(
                    (
                        multimodal.Text.create(
                            content=item["caption"],
                            uri=f"{item['pairid']}",
                            source=self.name,
                            language="en",
                        ),
                        multimodal.PackSequenceIndex(
                            sequence_id=sequence_id,
                            index="refined_caption",
                        ),
                    )
                )

        storager.add_texts(texts)


@dataclass
class SEEDEDIT(BaseIngestor):
    name: str = "20250908-seededit"
    vault_path: str = "/mnt/jfs/datasets/vault/edit/20250908-seededit"

    task_chunk_size: int = 2
    num_workers: int = 25

    root: str = "s3+b://liushiyu/daily_save_06/0612/seed_data/split_0002/"

    def prepare_tasks(self) -> list[Any]:
        tar_urls = expand_tar_urls([self.root])
        return tar_urls

    def process_tasks(
        self, tasks_chunk: list[Any], worker_id: int
    ) -> Generator[multimodal.PackSequence, None, None]:
        source = self.name

        for tar_url in tasks_chunk:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(tar_url),
                awds.atarfile_to_samples(handler=awds.print_exception_and_continue),
                wds.rename(
                    image=awds.IMAGE_EXTENSIONS_LISTSTR,
                    handler=awds.print_exception_and_continue,
                ),
                awds.map_dict(json=json.loads),
                awds.group_by(sample_to_group_id=lambda x, _: x["json"]["__group__"]),
            )

            for samples in tqdm(
                dataset,
                desc=f"{tar_url.replace(self.root, 'seededit')}",
                position=worker_id % 8 + 1,
                leave=True,
            ):
                samples: list[dict]

                __group__: str = samples[0]["json"]["__group__"]

                images = [
                    multimodal.Image.create(
                        sample["image"],
                        uri=f"{megfile.SmartPath(sample['__url__']).name}##{sample['__key__']}",
                        source=source,
                    )
                    for sample in samples
                ]
                texts = [
                    multimodal.Text.create(
                        content=txt,
                        uri=f"{__group__}_txt_{i}",
                        source=source,
                        language="en",
                    )
                    for i, txt in enumerate(samples[0]["json"]["caption"])
                ]

                items: list[multimodal.Image | multimodal.Text] = [images[0]]
                for txt, img in zip(texts, images[1:]):
                    items.append(txt)
                    items.append(img)

                yield multimodal.PackSequence.from_sequence(
                    sequence=items,
                    source=source,
                    uri=__group__,
                    meta=dict(tar_url=tar_url),
                )


@dataclass
class GeminiEcho4oGPT(BaseIngestor):
    name: str = "20250908-cref_gemini_echo4o"
    vault_path: str = "/mnt/jfs/datasets/vault/edit/20250908-cref_gemini_echo4o"

    task_chunk_size: int = 2
    num_workers: int = 24

    root: str = "s3+b://chengwei-base/data/cref_gemini_echo4o/"

    def prepare_tasks(self) -> list[Any]:
        tar_urls = expand_tar_urls(self.root)
        return tar_urls

    def process_tasks(
        self, tasks_chunk: list[Any], worker_id: int
    ) -> Generator[multimodal.PackSequence, None, None]:
        source = self.name

        for tar_url in tasks_chunk:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(tar_url),
                awds.atarfile_to_samples(handler=awds.print_exception_and_continue),
                awds.group_by(
                    sample_to_group_id=lambda x, _: x["__key__"].split("_")[0]
                ),
            )

            for samples in tqdm(
                dataset,
                desc=f"{tar_url.replace(self.root, self.name)}",
                position=worker_id % 8 + 1,
                leave=True,
            ):
                samples: list[dict]

                __group__: str = f"{megfile.SmartPath(tar_url).stem}_{samples[0]['__key__'].split('_')[0]}"

                images = []
                texts = []
                for sample in samples:
                    if "caption" in sample["__key__"]:
                        texts.append(
                            (
                                multimodal.Text.create(
                                    content=json.loads(sample["json"])["caption"],
                                    uri=sample["__key__"],
                                    source=source,
                                    language="en",
                                ),
                                "caption",
                            )
                        )
                    else:
                        images.append(
                            (
                                multimodal.Image.create(
                                    PIL.Image.open(
                                        io.BytesIO(sample["png"])
                                    ),  # 数据都是png格式，太占地方，传入pil image，会自动转成webp
                                    uri=sample["__key__"],
                                    source=source,
                                ),
                                f"{sample['__key__'].split('_', maxsplit=1)[1]}",
                            )
                        )

                yield multimodal.PackSequence.create(
                    images=images,
                    texts=texts,
                    source=source,
                    uri=__group__,
                    meta=dict(tar_url=tar_url),
                )


if __name__ == "__main__":
    import fire

    fire.Fire(
        dict(
            echo_gpt_4o=EchoGPT4o,
            xp_250901=QwenTextualImage3,
            mj=MJTextualImage,
            cirr=CIRR,
            seededit=SEEDEDIT,
            cref_gemini_echo4o=GeminiEcho4oGPT,
        )
    )

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import duckdb
import megfile
import toml
from pydantic import BaseModel

from vault.backend.duckdb import DistributedDuckDBWriter, DuckDBHandler
from vault.storage.lanceduck.multimodal import ID, MultiModalStorager
from vault.utils import batched
from vault.utils.ingest import run_concurrently

GET_SEQUENCES_SQL = """
-- 这个查询用于根据一个或多个 sequence_id 高效地获取所有相关的图片和文本信息。
-- 它利用 DuckDB 的 list() 聚合函数和 struct 功能，为每个序列ID返回一行结果，
-- 其中包含一个图片对象列表和一个文本对象列表。

SELECT
    s.id AS sequence_id,
    s.uri AS uri,
    s.source AS source,
    s.meta AS meta,
    -- 聚合所有与该序列关联的图片信息。
    -- [修正] 使用 list(DISTINCT ...) 来防止因 JOIN 产生的重复项。
    -- 当一个 sequence 关联了多张图片和多段文本时，JOIN 会产生笛卡尔积，
    -- 如果不使用 DISTINCT，图片和文本都会在列表中重复出现。
    list(DISTINCT {
        'id': i.id,
        'uri': i.uri,
        'source': i.source,
        'width': i.width,
        'height': i.height,
        'index': si."index"
    }) FILTER (WHERE i.id IS NOT NULL) AS images,

    -- 同样地，聚合所有与该序列关联的文本信息。
    -- [修正] 使用 list(DISTINCT ...) 来防止文本信息重复。
    list(DISTINCT {
        'id': t.id,
        'content': t.content,
        'uri': t.uri,
        'source': t.source,
        'language': t.language,
        'index': st."index"
    }) FILTER (WHERE t.id IS NOT NULL) AS texts

FROM
    sequences AS s
    -- 使用 LEFT JOIN 来确保即使序列只有图片或只有文本（或都没有），它仍然会出现在结果中。
    -- 如果用 INNER JOIN，那么没有图片或没有文本的序列将被过滤掉。
    LEFT JOIN sequence_images AS si ON s.id = si.sequence_id
    LEFT JOIN images AS i ON si.image_id = i.id
    LEFT JOIN sequence_texts AS st ON s.id = st.sequence_id
    LEFT JOIN texts AS t ON st.text_id = t.id
WHERE
    -- 使用 IN 子句来一次性查询多个 sequence_id。
    -- 在实际使用中，('seq_id_1', 'seq_id_2', ...) 会被具体的UUID列表替换。
    s.id IN ?
GROUP BY
    -- 按 sequence_id 分组，以便 list() 函数为每个序列聚合其对应的图片和文本。
    s.id, s.uri, s.source, s.meta
ORDER BY
    s.id
"""


now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Image(BaseModel):
    id: UUID
    width: int
    height: int
    index: str


class Text(BaseModel):
    id: UUID
    content: str
    language: Literal["cn", "en"] | None
    index: str


class ModalInstance(BaseModel):
    type: Literal["image", "text"]
    index: str
    require_loss: bool
    ref_index: str | None = None


class SequenceChoice(BaseModel):
    choice: list[ModalInstance]


class PackSequence(BaseModel):
    sequence_id: UUID
    create_time: str
    uri: str
    source: str
    task_type: str
    images: list[Image]
    texts: list[Text]
    sequence_choices: list[SequenceChoice] | None = None
    choices_weights: list[float] | None = None

    def __post_init__(self):
        assert len(self.images) == len({x.index for x in self.images})
        assert len(self.texts) == len({x.index for x in self.texts})

    def get_image(self, index: str) -> Image | None:
        for image in self.images:
            if image.index == index:
                return image
        return None

    def get_text(self, index: str) -> Text | None:
        for text in self.texts:
            if text.index == index:
                return text
        return None

    @classmethod
    def from_sample(cls, sample: tuple, task_type: str):
        sequence_id, uri, source, images, texts = sample
        images = [Image(**x) for x in images]
        texts = [Text(**x) for x in texts]
        return cls(
            sequence_id=sequence_id,
            create_time=now,
            uri=uri,
            source=source,
            task_type=task_type,
            images=images,
            texts=texts,
        )


def model_to_list(obj: Any, is_root=True):
    """递归把 BaseModel 或容器转为 list"""
    if isinstance(obj, BaseModel):
        # 按字段定义顺序取值
        if is_root:
            return [
                model_to_list(getattr(obj, field), False)
                for field in obj.__class__.model_fields
            ]
        else:
            return obj.model_dump()
    elif isinstance(obj, list):
        return [model_to_list(item, False) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(model_to_list(item, False) for item in obj)
    else:
        return obj


class TrainDataExtractor:
    vault_path: str
    task_type: Literal["t2i", "edit"] | dict[str, Literal["t2i", "edit"]]
    storager: MultiModalStorager

    def __init__(
        self,
        vault_path,
        task_type,
        extract_output_path: str,
        allowed_source: list[str] | None = None,
    ):
        self.vault_path = vault_path
        self.allowed_source = allowed_source

        vault_id = hashlib.sha256(self.vault_path.encode("utf-8")).digest()
        self.vault_id = int.from_bytes(vault_id, "big")

        self.task_type = task_type
        self.storager = MultiModalStorager(vault_path, read_only=True)
        self.extract_output_path = extract_output_path
        self.handler = DuckDBHandler(
            megfile.smart_load_text(
                megfile.smart_path_join(
                    Path(__file__).resolve().parent, "sequence_schema.sql"
                )
            ),
            os.path.join(self.extract_output_path, "train.db"),
            read_only=False,
        )

    def _worker_save(self, chunk_sequence_id, worker_id: int):
        handler = DistributedDuckDBWriter(self.handler).get_worker_handler()

        storager = MultiModalStorager(self.vault_path, read_only=True)
        results = storager.meta_handler.conn.execute(
            GET_SEQUENCES_SQL,
            [tuple(chunk_sequence_id)],
        ).fetchall()

        if not isinstance(self.task_type, dict):
            _task_types = [self.task_type for x in results]
        else:
            _task_types = [self.task_type[x] for x in results]

        _results = []
        for x in zip(results, _task_types):
            s = self.set_sequence_choices(*x)
            if s is not None:
                _results.append(model_to_list(s))

        insert_sql = f"""
            INSERT INTO sequences_data (sequence_id, vault_path, create_time, uri, source, task_type, images, texts, sequence_choices, choices_weights)
            VALUES
                (?, '{self.vault_path}', ?, ?, ?, ?, ?, ?, ?, ?);
        """  # noqa: E501

        con = handler.conn
        con.begin()
        con.executemany(insert_sql, _results)
        con.commit()
        con.close()

    def save(self, commit: bool = True):
        if not os.path.exists(self.extract_output_path):
            os.makedirs(self.extract_output_path, exist_ok=True)

        with duckdb.connect(str(self.storager.metadata_path), read_only=True) as conn:
            sources = conn.execute("SELECT DISTINCT source FROM sequences").fetchall()
            placeholders = ",".join(["?" for _ in sources])
            query = f"SELECT id FROM sequences WHERE source IN ({placeholders})"
            sources = [tuple(x[0] for x in sources)]

            if isinstance(self.task_type, dict):
                assert all(x in sources[0] for x in self.task_type)

            result = conn.execute(query, *sources).fetchall()
            assert conn.description is not None
            column_names = [desc[0] for desc in conn.description]
            output = [dict(zip(column_names, row)) for row in result]
            sequence_id_list = tuple(ID.from_(s["id"]).to_uuid() for s in output)

        chunked_sequence_id_list = list(batched(sequence_id_list, 1800))
        run_concurrently(self._worker_save, chunked_sequence_id_list, 8)

        if commit:
            DistributedDuckDBWriter(self.handler).commit()

    def find_target_caption(self, sequence_id, uri, source, meta, images, texts):
        try:
            _meta_info = None
            for text in texts:
                if text.index == "meta_info":
                    _meta_info = json.loads(text.content)
                    break
            if _meta_info is not None:
                return _meta_info["independent_captions"]["target"]

            _r = self.storager.meta_handler.query_batch(
                "select value_json from sample_annotations where sequence_id = ? and name = ?",
                [
                    ID.from_(sequence_id).to_uuid(),
                    "describe_differences_20251015_qwen3-vl-30ba3b",
                ],
            )
            if len(_r) > 0:
                sample_annotation = json.loads(_r[0]["value_json"])
                target_caption = sample_annotation["independent_captions"]["target"]
                return target_caption

            if meta is not None:
                meta = json.loads(meta)
                if "independent_captions" in meta:
                    return meta["independent_captions"]["P2_target_after"]
        except Exception:
            pass

        return None

    def set_sequence_choices(self, sample: tuple, task_type: str):
        sequence_id, uri, source, meta, images, texts = sample

        if self.allowed_source is not None and source not in self.allowed_source:
            return None

        images = [Image(**x) for x in images]
        texts = [Text(**x) for x in texts]

        target_caption = self.find_target_caption(
            sequence_id, uri, source, meta, images, texts
        )

        if target_caption is not None:
            texts.append(
                Text(
                    id=ID.hash(target_caption).to_uuid(),
                    content=f" 目标图描述是: {target_caption}",
                    language="cn",
                    index="captions/target",
                )
            )

        sequence_choices = []

        allowed_instruction_indices = {text.index for text in texts}

        candidate_instruction_indices = [
            "instruction_cn",
            "instruction_en",
            "primary_description_cn",
            "primary_description_en",
            "sample_description_cn",
            "sample_description_en",
        ]

        if target_caption is not None:
            text_indices_choices = [
                (cii, "captions/target")
                for cii in candidate_instruction_indices
                if cii in allowed_instruction_indices
            ]
        else:
            text_indices_choices = [
                (cii,)
                for cii in candidate_instruction_indices
                if cii in allowed_instruction_indices
            ]

        if not {"source", "target"}.issubset({image.index for image in images}):
            return None

        for text_indices in text_indices_choices:
            sequence_choice = [
                ModalInstance(
                    type="image",
                    index="source",
                    ref_index=None,
                    require_loss=False,
                )
            ]
            for text_index in text_indices:
                sequence_choice.append(
                    ModalInstance(
                        type="text",
                        index=text_index,
                        ref_index=None,
                        require_loss=False,
                    )
                )
            sequence_choice.append(
                ModalInstance(
                    type="image",
                    index="target",
                    ref_index=None,
                    require_loss=True,
                )
            )

            if set(text_indices).issubset({text.index for text in texts}):
                sequence_choices.append(SequenceChoice(choice=sequence_choice))

        allowed_text_indices = []
        for sc in sequence_choices:
            for choice in sc.choice:
                if choice.type == "text":
                    allowed_text_indices.append(choice.index)

        allowed_image_indices = []
        for sc in sequence_choices:
            for choice in sc.choice:
                if choice.type == "image":
                    allowed_image_indices.append(choice.index)

        images = [x for x in images if x.index in allowed_image_indices]
        texts = [x for x in texts if x.index in allowed_text_indices]

        assert len(sequence_choices) > 0, (
            f"{sequence_id=} {uri=} {source=} {task_type=} {images=} {texts=}"
        )

        sequence = PackSequence(
            sequence_id=sequence_id,
            create_time=now,
            uri=uri,
            source=source,
            task_type=task_type,
            images=images,
            texts=texts,
        )

        sequence.sequence_choices = sequence_choices
        return sequence


def find_vault_paths(root_folder: str = "/mnt/marmot") -> list[str]:
    """
    递归查找一个文件夹下的所有满足特定条件的文件夹（保险库）。

    一个文件夹被识别为“保险库”需要满足以下两个条件：
    1. 包含一个名为 'metadata.duckdb' 的文件。
    2. 包含一个名为 'images' 的文件夹。

    当找到一个“保险库”文件夹后，将不再继续递归查找其子文件夹。

    参数:
        root_folder (str): 需要开始搜索的根文件夹路径。

    返回:
        list: 所有找到的“保险库”文件夹的路径列表。
    """
    vault_paths = []
    # os.walk() 会返回一个三元组 (当前路径, [子目录], [子文件])
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # 检查 'metadata.duckdb' 文件是否存在
        has_metadata = "metadata.duckdb" in filenames
        # 检查 'images' 文件夹是否存在
        has_images_dir = "images" in dirnames

        if has_metadata and has_images_dir:
            vault_paths.append(dirpath)
            # 清空子目录列表，防止 os.walk() 继续递归这个文件夹
            dirnames[:] = []
    return vault_paths


def load_toml(file_path: str | Path) -> dict:
    data = toml.load(str(file_path))

    # 处理 datetime 类型的转换
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    data["updated_at"] = datetime.fromisoformat(data["updated_at"])

    if "index_descriptions" in data:
        index_descriptions = {}
        for index_str, description in data["index_descriptions"].items():
            index_type, index_name = index_str.split(":")
            index_descriptions[(index_type, index_name)] = description
        data["index_descriptions"] = index_descriptions

    return data


def is_valid_vault(vault_path: str) -> bool:
    # 加载 vault.toml 文件（如果需要的话）
    v = load_toml(Path(vault_path) / "vault.toml")
    return not (
        "推荐用于预训练" in v["tags"]
        or "弃用" in v["tags"]
        or "xingpeng" in vault_path
        or "examples" in vault_path
        or "i-liushiyu" not in vault_path
    )


def main(
    task_type: str = "edit",
    extract_output_path: str = "/data/stepflow_zhibo_20251104/",
):
    # vault_paths = find_vault_paths()
    # handler = DuckDBHandler(
    #     megfile.smart_load_text(
    #         megfile.smart_path_join(
    #             Path(__file__).resolve().parent, "sequence_schema.sql"
    #         )
    #     ),
    #     "/tmp/stepflow_zhibo_20251103/train.db",
    #     read_only=False,
    # )
    # allowed_source = [
    #     x["source"]
    #     for x in handler.query_batch("SELECT DISTINCT source FROM sequences_data")
    # ]
    # vault_paths = ["/mnt/marmot/i-liushiyu/hq_v2"]
    vs = """
    /mnt/marmot/i-liushiyu/pretrain_spatialvid_new
    /mnt/marmot/sirui/ScreenMusings-251022
    /mnt/marmot/i-liushiyu/motion_0703_9885_reverse
    /mnt/marmot/i-liushiyu/motion_0709_pj10138
    /mnt/marmot/i-liushiyu/motion_0703_9885
    /mnt/marmot/i-liushiyu/motion_0902_new_reverse
    /mnt/marmot/i-liushiyu/hq_v2
    /mnt/marmot/i-liushiyu/motion_0709_pj10138_reverse
    /mnt/marmot/i-liushiyu/motion_1010
    /mnt/marmot/i-liushiyu/motion_0902_new
    /mnt/marmot/yuchenghan/20251010_000_scene_distill_correct_ratio_new_template
    /mnt/marmot/i-liushiyu/motion_liaojie
    /mnt/marmot/i-liushiyu/motion_liaojie_reverse
    """
    vault_paths = [v.strip() for v in vs.strip().split("\n")]
    vault_paths = ["/mnt/marmot/sirui/ScreenMusings-251022"]

    allowed_source = None

    for vault_path in vault_paths:
        if "#" in vault_path:
            continue
        # if not is_valid_vault(vault_path):
        #     print(f"skipping {vault_path}")
        #     continue

        print(f"processing {vault_path}")

        extractor = TrainDataExtractor(
            vault_path, task_type, extract_output_path, allowed_source
        )
        extractor.save(commit=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

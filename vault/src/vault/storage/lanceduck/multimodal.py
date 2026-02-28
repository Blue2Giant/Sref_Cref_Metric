import importlib.resources as pkg_resources
import traceback
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import megfile
import numpy as np
import PIL.Image
import pyarrow as pa
import xxhash
from loguru import logger

from vault.backend.duckdb import DistributedDuckDBWriter, DuckDBHandler
from vault.backend.lance import DistributedLanceWriter, LanceItem, LanceTaker
from vault.schema.multimodal import (
    ID,
    Annotation,
    Image,
    PackSequence,
    PackSequenceIndex,
    SampleAnnotation,
    Text,
)
from vault.utils import batched, jsonify_meta
from vault.utils.image import image_edge_characteristics, image_entropy
from vault.utils.pqh_hash import pdq_hasher


@dataclass(frozen=True)
class LanceAnnotation(LanceItem):
    id: bytes
    name: str
    type_: str | None
    creator_name: str | None
    blob: bytes | None = None
    meta: str | None = None

    @classmethod
    def from_annotation(cls, anno: Annotation):
        return cls(
            id=anno.id.to_bytes(),
            name=anno.name,
            type_=anno.type_,
            creator_name=getattr(anno.creator, "name", None),
            blob=anno.blob,
        )

    @classmethod
    def get_schema(cls):
        return pa.schema(
            [
                pa.field("id", pa.uuid(), nullable=False),
                pa.field("name", pa.string()),
                pa.field("type_", pa.string()),
                pa.field("creator_name", pa.string()),
                pa.field("blob", pa.binary()),
                pa.field("meta", pa.string()),
            ]
        )


@dataclass(frozen=True)
class LanceImage(LanceItem):
    id: bytes
    image: bytes

    # 基础元信息
    uri: str
    source: str

    # 图像元数据
    file_hash: bytes
    file_size: int
    width: int
    height: int
    aspect_ratio: float
    color_mode: str

    # 统计特征
    mean_saturation: float
    mean_lightness: float

    # 质量特征
    clarity: float  # 拉普拉斯算子方差, 值越小越模糊
    entropy: float  # 信息熵
    edge_probability: float
    edge_near_patch_min_std: float

    # PDQ HASH, pHash升级版
    pdq_hash: bytes | None
    pdq_quality: float  # 值越高，图片内容越丰富

    @classmethod
    def from_image(cls, img: Image):
        image = img.pil_image

        image_bytes = img.blob

        image_512x512_rgb = image.resize(
            (512, 512), resample=PIL.Image.Resampling.BILINEAR
        )
        image_512x512_l = image_512x512_rgb.convert("L")
        image_512x512_l_np = np.array(image_512x512_l)

        pdq_quality, pdq_hash = pdq_hasher(image_512x512_l)

        mean_lightness = float(np.mean(image_512x512_l_np))
        mean_saturation = float(
            np.mean(np.array(image_512x512_rgb.convert("HSV"))[:, :, 1]) / 255.0
        )

        variance = float(cv2.Laplacian(image_512x512_l_np, cv2.CV_64F).var())

        edge_probability, edge_near_patch_min_std = image_edge_characteristics(
            image_512x512_l_np
        )

        return cls(
            id=img.id.to_bytes(),
            image=image_bytes,
            uri=img.uri,
            source=img.source,
            file_hash=xxhash.xxh3_128_digest(image_bytes),
            file_size=len(image_bytes),
            width=image.width,
            height=image.height,
            aspect_ratio=image.width / image.height,
            color_mode=image.mode,
            mean_saturation=mean_saturation,
            mean_lightness=mean_lightness,
            clarity=variance,
            entropy=image_entropy(image_512x512_l_np),
            edge_probability=edge_probability,
            edge_near_patch_min_std=edge_near_patch_min_std,
            pdq_quality=pdq_quality,
            pdq_hash=pdq_hash,
        )

    @classmethod
    def get_schema(cls):
        return pa.schema(
            [
                pa.field("id", pa.uuid(), nullable=False),
                pa.field("image", pa.binary(), nullable=False),
                # 基础元信息
                pa.field("uri", pa.string()),
                pa.field("source", pa.string()),
                # 图像元数据
                pa.field("file_hash", pa.binary(16)),
                pa.field("file_size", pa.int64()),
                pa.field("width", pa.int64()),
                pa.field("height", pa.int64()),
                pa.field("aspect_ratio", pa.float32()),
                pa.field("color_mode", pa.string()),
                # 统计特征
                pa.field("mean_saturation", pa.float32()),
                pa.field("mean_lightness", pa.float32()),
                # 质量特征
                pa.field("clarity", pa.float32()),  # variance_of_laplacian
                pa.field("entropy", pa.float32()),  # 信息熵
                pa.field("edge_probability", pa.float32()),
                pa.field("edge_near_patch_min_std", pa.float32()),
                # PDQ HASH, pHash升级版
                pa.field("pdq_hash", pa.binary(32)),
                pa.field("pdq_quality", pa.float32()),  # 值越高，图片内容越丰富
            ]
        )


def load_duckdb_sql(path: str):
    with pkg_resources.as_file(
        pkg_resources.files(__package__).joinpath(path)
    ) as sql_path:
        return megfile.smart_load_text(sql_path)


def convert_as_id(data: Any) -> Any:
    """
    递归地遍历数据结构（字典或列表），
    将所有 key 为 'id' 或以 '_id' 结尾的字符串值转换为 ID 对象。
    """
    import uuid

    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # 检查 key 是否匹配条件，并且 value 是一个字符串
            if (
                key.endswith("_id")
                or key == "id"
                and isinstance(value, (str, bytes, uuid.UUID))
            ):
                try:
                    # 尝试将字符串转换为 ID
                    new_dict[key] = ID.from_(value)
                except ValueError:
                    # 如果转换失败（例如，不是有效的ID字符串），则保留原值
                    new_dict[key] = value
            else:
                # 如果 key 不匹配，则对 value 进行递归调用
                new_dict[key] = convert_as_id(value)
        return new_dict
    elif isinstance(data, list):
        # 如果是列表，对每个元素进行递归调用
        return [convert_as_id(element) for element in data]
    else:
        # 对于所有其他类型（int, float, None, bool 等），直接返回原值
        return data


class MultiModalStorager:
    LANCE_SCHEMA: dict = dict(
        images=LanceImage.get_schema(), annotations=LanceAnnotation.get_schema()
    )

    DUCKDB_SCHEMA: str = load_duckdb_sql("sql/schema.sql") + load_duckdb_sql(
        "sql/sample_annotations_schema.sql"
    )

    _DUCKDB_TABLE_NAME_ORDER = [
        "creators",
        "images",
        "texts",
        "sequences",
        "sequence_images",
        "sequence_texts",
        "annotations",
        "image_annotations",
        "text_annotations",
        "sample_annotations",
        "sample_annotation_elements",
    ]

    def __init__(
        self,
        path: str | Path | megfile.SmartPath,
        read_only: bool = True,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.path = megfile.SmartPath(path)
        self.read_only = read_only
        self.metadata_path = metadata_path or str(self.path / "metadata.duckdb")
        self.meta_handler = DuckDBHandler(
            schema=self.DUCKDB_SCHEMA,
            read_only=self.read_only,
            db_path=self.metadata_path,
        )

        self.lance_uris = {n: str(self.path / n) for n in self.LANCE_SCHEMA}

        self.lance_taker = LanceTaker()

    @classmethod
    def init(cls, path: str, metadata_path: str | Path | None = None):
        megfile.SmartPath(path).mkdir(exist_ok=True, parents=True)

        with DuckDBHandler(
            schema=cls.DUCKDB_SCHEMA,
            db_path=metadata_path or str(megfile.SmartPath(path) / "metadata.duckdb"),
        ) as duckdb_handler:
            duckdb_handler.create()

    def schema_summary(self) -> str:
        """
        Returns a human-readable summary of the storage schemas.
        """
        # Header for the overall representation
        header = f"{self.__class__.__name__} Schema Summary\n"
        header += f"Storage Path: {self.path}\n"
        header += "=" * 80 + "\n\n"

        # Design Philosophy Section
        design_philosophy = "DESIGN PHILOSOPHY\n"
        design_philosophy += "-" * 50 + "\n"
        design_philosophy += "  Hybrid Storage Architecture:\n"
        design_philosophy += "     • Lance: Columnar format with random access for binary data (images, annotations)\n"
        design_philosophy += (
            "     • DuckDB: Relational metadata with ACID compliance and SQL queries\n"
        )
        design_philosophy += "     • Separation of concerns: Data vs Metadata\n\n"

        design_philosophy += "  Multi-Modal Data Model:\n"
        design_philosophy += (
            "     • Sequences: Logical containers for related content\n"
        )
        design_philosophy += "     • Images & Texts: Core content with rich metadata\n"
        design_philosophy += (
            "     • Annotations: Flexible labeling system with creator tracking\n"
        )
        design_philosophy += "     • Index-based ordering: Preserves content sequence and relationships\n\n"

        design_philosophy += "  Performance Optimizations:\n"
        design_philosophy += (
            "     • Distributed writing: Concurrent processing with atomic commits\n"
        )
        design_philosophy += "     • Image quality metrics: Clarity, entropy, edge detection for filtering\n"
        design_philosophy += (
            "     • PDQ hashing: Advanced perceptual hashing for duplicate detection\n"
        )
        design_philosophy += (
            "     • Strategic indexing: Optimized for common query patterns\n\n"
        )

        # --- Format LANCE_SCHEMA ---
        lance_header = "LANCE SCHEMA (PyArrow Tables)\n"
        lance_header += "-" * 50 + "\n"
        lance_header += "  Purpose: High-performance columnar storage with random access for binary content\n"
        lance_header += (
            "  Technology: Lance format with PyArrow for columnar efficiency\n"
        )
        lance_header += (
            "  Content: Raw images, annotation blobs, and computed features\n\n"
        )

        lance_body_parts = []
        for i, (table_name, schema) in enumerate(self.LANCE_SCHEMA.items(), 1):
            # Format table information with clear structure
            table_info = f"  {i}. Table: {table_name}\n"
            table_info += f"     URI: {self.lance_uris[table_name]}\n"

            # Add table-specific design notes
            if table_name == "images":
                table_info += (
                    "     Design: Stores raw image bytes + computed quality metrics\n"
                )
                table_info += (
                    "     Features: PDQ hash, clarity, entropy, edge detection\n"
                )
                table_info += (
                    "     Processing: 512x512 RGB conversion for consistent analysis\n"
                )
            elif table_name == "annotations":
                table_info += (
                    "     Design: Flexible annotation system with creator tracking\n"
                )
                table_info += (
                    "     Content: Structured labels, binary blobs, metadata\n"
                )
                table_info += (
                    "     Relations: Links to images/texts via junction tables\n"
                )

            table_info += "     Schema:\n"

            # Format the pyarrow schema with proper indentation
            schema_str = str(schema)
            indented_schema = "\n".join(
                f"        {line}" for line in schema_str.split("\n")
            )
            table_info += f"{indented_schema}\n"

            lance_body_parts.append(table_info)

        lance_body = "\n".join(lance_body_parts) + "\n\n"

        # --- Format DUCKDB_SCHEMA ---
        duckdb_header = "DUCKDB SCHEMA (SQL Tables)\n"
        duckdb_header += "-" * 50 + "\n"
        duckdb_header += (
            "  Purpose: Relational metadata with ACID compliance and SQL queries\n"
        )
        duckdb_header += (
            "  Technology: DuckDB for analytical workloads and complex joins\n"
        )
        duckdb_header += (
            "  Content: Structured metadata, relationships, and searchable attributes\n"
        )
        duckdb_header += "  Design: Normalized schema with junction tables for many-to-many relationships\n\n"

        table_names = DistributedDuckDBWriter._parse_table_names(self.DUCKDB_SCHEMA)

        # Format table list with design notes
        table_list = "  Tables:\n"
        table_design_notes = {
            "creators": "User/creator management with metadata",
            "annotations": "Flexible labeling system with creator tracking",
            "images": "Image metadata (dimensions, source, URI)",
            "texts": "Text content with language and source tracking",
            "sequences": "Logical containers for related content groups",
            "image_annotations": "Many-to-many: Images ↔ Annotations",
            "text_annotations": "Many-to-many: Texts ↔ Annotations",
            "sequence_images": "Many-to-many: Sequences ↔ Images (with ordering)",
            "sequence_texts": "Many-to-many: Sequences ↔ Texts (with ordering)",
        }

        for i, table_name in enumerate(table_names, 1):
            design_note = table_design_notes.get(table_name, "Data table")
            table_list += f"    {i}. {table_name} - {design_note}\n"
        table_list += "\n"

        # Then show the clean SQL schema
        sql_header = "  SQL Schema:\n"
        sql_lines = self.DUCKDB_SCHEMA.strip().split("\n")
        formatted_sql = []

        for line in sql_lines:
            line = line.strip()
            if not line:
                # Keep empty lines for readability
                formatted_sql.append("")
            elif line.startswith("--"):
                # Format comments with proper indentation
                formatted_sql.append(f"    {line}")
            else:
                # Format SQL statements with proper indentation
                formatted_sql.append(f"    {line}")

        duckdb_body = table_list + sql_header + "\n".join(formatted_sql) + "\n\n"

        # Summary footer
        footer = "=" * 80 + "\n"
        footer += f"Summary: {len(self.LANCE_SCHEMA)} Lance tables, {len(table_names)} DuckDB tables\n"
        footer += "Storage: Lance format for binary data, DuckDB for metadata\n"

        return f"{header}{design_philosophy}{lance_header}{lance_body}{duckdb_header}{duckdb_body}{footer}"

    @property
    def sequence_sources(self) -> list[str]:
        sources = self.meta_handler.query_batch("SELECT DISTINCT source FROM sequences")
        source_list = [s["source"] for s in sources if s["source"]]
        return source_list

    def stat_sequence(self) -> list[dict]:
        return self.meta_handler.query_batch(load_duckdb_sql("sql/stat_sequences.sql"))

    def stat_text(self) -> list[dict]:
        return self.meta_handler.query_batch(load_duckdb_sql("sql/stat_texts.sql"))

    def stat_image(self) -> list[dict]:
        return self.meta_handler.query_batch(load_duckdb_sql("sql/stat_images.sql"))

    @staticmethod
    def _as_string_index(index: int | str) -> str:
        if isinstance(index, int):
            index = f"index_{index}"
        return index

    def _process_annotation(
        self, anno: Annotation, duckdb_data: defaultdict, lance_annotations: list
    ):
        if anno.blob is not None:
            lance_annotations.append(LanceAnnotation.from_annotation(anno))

        creator_id = None
        if anno.creator is not None:
            creator_id = anno.creator.id
            duckdb_data["creators"].append(
                dict(
                    id=creator_id,
                    name=anno.creator.name,
                    meta=anno.creator.json_meta,
                )
            )
        duckdb_data["annotations"].append(
            dict(
                id=anno.id.to_uuid(),
                name=anno.name,
                type=anno.type_,
                creator_id=creator_id,
                meta=anno.json_meta,
            )
        )
        return anno.id.to_uuid()

    def _convert_sequence(self, s: PackSequence):
        duckdb_data = defaultdict(list)
        lance_images = []
        lance_annotations = []

        duckdb_data["sequences"].append(
            dict(
                id=s.id.to_uuid(),
                uri=s.uri,
                source=s.source,
                meta=s.json_meta,
            )
        )

        for img, img_index in s.images:
            img: Image
            lance_image = LanceImage.from_image(img)

            lance_images.append(lance_image)

            duckdb_data["images"].append(
                dict(
                    id=img.id.to_uuid(),
                    uri=img.uri,
                    source=img.source,
                    width=lance_image.width,
                    height=lance_image.height,
                )
            )

            duckdb_data["sequence_images"].append(
                dict(
                    sequence_id=s.id.to_uuid(),
                    image_id=img.id.to_uuid(),
                    index=self._as_string_index(img_index),
                )
            )

            if img.annotations is not None:
                for anno in img.annotations:
                    annotation_id = self._process_annotation(
                        anno, duckdb_data, lance_annotations
                    )
                    duckdb_data["image_annotations"].append(
                        dict(
                            image_id=img.id.to_uuid(),
                            annotation_id=annotation_id,
                        )
                    )

        for txt, txt_index in s.texts:
            duckdb_data["texts"].append(
                dict(
                    id=txt.id.to_uuid(),
                    content=txt.content,
                    uri=txt.uri,
                    source=txt.source,
                    language=txt.language,
                )
            )

            duckdb_data["sequence_texts"].append(
                dict(
                    sequence_id=s.id.to_uuid(),
                    text_id=txt.id.to_uuid(),
                    index=self._as_string_index(txt_index),
                )
            )

            if txt.annotations is not None:
                for anno in txt.annotations:
                    annotation_id = self._process_annotation(
                        anno, duckdb_data, lance_annotations
                    )
                    duckdb_data["text_annotations"].append(
                        dict(
                            text_id=txt.id.to_uuid(),
                            annotation_id=annotation_id,
                        )
                    )

        return dict(
            duckdb=duckdb_data,
            lance=dict(
                images=LanceImage.to_batch(lance_images),
                annotations=LanceAnnotation.to_batch(lance_annotations),
            ),
        )

    @staticmethod
    def _merge_batch(batch, lance_names: list[str]):
        out = dict(duckdb=defaultdict(list), lance=dict())

        for name in lance_names:
            tables = [
                b["lance"][name]
                for b in batch
                if name in b["lance"] and b["lance"][name] is not None
            ]
            if tables:
                out["lance"][name] = pa.Table.from_batches(tables)

        for item in batch:
            for k, v in item["duckdb"].items():
                out["duckdb"][k].extend(v)
        return out

    @classmethod
    def _batch_insert_duckdb(
        cls, handler: DuckDBHandler, duckdb_data: dict[str, list[dict]]
    ):
        _sqls = []
        _names = []
        for tn in cls._DUCKDB_TABLE_NAME_ORDER:
            if tn in duckdb_data:
                _sqls.append(f"INSERT OR REPLACE INTO {tn} SELECT * FROM df")
                _names.append(tn)

        handler.add_multiply(duckdb_data, _sqls, _names)

    def add_sequences(self, sequences: Iterable[PackSequence], batch_size: int = 5000):
        handler = DistributedDuckDBWriter(
            handler=self.meta_handler, table_names=self._DUCKDB_TABLE_NAME_ORDER
        ).get_worker_handler()

        lance_tables = {
            name: DistributedLanceWriter(self.lance_uris[name], schema=schema)
            for name, schema in self.LANCE_SCHEMA.items()
        }

        for items in batched(
            map(self._convert_sequence, sequences), batch_size=batch_size
        ):
            batch = self._merge_batch(items, lance_names=list(self.LANCE_SCHEMA.keys()))

            try:
                for name, table in batch["lance"].items():
                    table: pa.Table
                    if table is not None and not table.num_rows == 0:
                        lance_tables[name].write_batch(table)

                self._batch_insert_duckdb(handler, batch["duckdb"])
            except Exception:
                logger.error(f"{traceback.format_exc()}")

    def add_texts(
        self,
        texts: Iterable[tuple[Text, PackSequenceIndex] | Text],
        batch_size: int = 10000,
        duckdb_handler: DuckDBHandler | None = None,
    ):
        """
        批量添加文本，支持孤立添加和序列绑定两种模式

        Args:
            texts: 文本数据迭代器，每个元素为：
                - Text: 孤立添加，只添加文本本身
                - (Text, PackSequenceIndex): 添加文本并绑定到序列
            batch_size: 批处理大小
        """

        def convert_texts_batch(texts_batch):
            duckdb_data = defaultdict(list)
            lance_annotations = []

            for text_item in texts_batch:
                if isinstance(text_item, tuple):
                    # 序列绑定模式: (Text, PackSequenceIndex)
                    txt, pack_index = text_item
                    bind_to_sequence = True
                    index = pack_index.index
                    sequence_id = pack_index.sequence_id
                else:
                    # 孤立添加模式: Text
                    txt = text_item
                    bind_to_sequence = False
                    index = None
                    sequence_id = None

                # 添加文本到 texts 表
                duckdb_data["texts"].append(
                    dict(
                        id=txt.id.to_uuid(),
                        content=txt.content,
                        uri=txt.uri,
                        source=txt.source,
                        language=txt.language,
                    )
                )

                # 如果需要绑定到序列，添加到 sequence_texts 表
                if bind_to_sequence and sequence_id is not None and index is not None:
                    duckdb_data["sequence_texts"].append(
                        dict(
                            sequence_id=sequence_id.to_uuid(),
                            text_id=txt.id.to_uuid(),
                            index=self._as_string_index(index),
                        )
                    )

                # 处理注释
                if txt.annotations is not None:
                    for anno in txt.annotations:
                        annotation_id = self._process_annotation(
                            anno, duckdb_data, lance_annotations
                        )
                        duckdb_data["text_annotations"].append(
                            dict(
                                text_id=txt.id.to_uuid(),
                                annotation_id=annotation_id,
                            )
                        )

            return dict(
                duckdb=duckdb_data,
                lance=dict(
                    annotations=LanceAnnotation.to_batch(lance_annotations)
                    if lance_annotations
                    else None,
                ),
            )

        handler = duckdb_handler or self.meta_handler
        if handler is None:
            raise ValueError("duckdb_handler is required")

        lance_tables = {
            name: DistributedLanceWriter(self.lance_uris[name], schema=schema)
            for name, schema in self.LANCE_SCHEMA.items()
        }

        for items in batched(
            map(convert_texts_batch, batched(texts, batch_size)), batch_size=1
        ):
            batch = self._merge_batch(items, lance_names=["annotations"])

            try:
                for name, table in batch["lance"].items():
                    table: pa.Table
                    if table is not None and not table.num_rows == 0:
                        lance_tables[name].write_batch(table)

                self._batch_insert_duckdb(handler, batch["duckdb"])
            except Exception:
                logger.error(f"{traceback.format_exc()}")

    def add_images(
        self,
        images: Iterable[tuple[Image, PackSequenceIndex] | Image],
        batch_size: int = 5000,
        duckdb_handler: DuckDBHandler | None = None,
    ):
        """
        批量添加图像，支持孤立添加和序列绑定两种模式

        Args:
            images: 图像数据迭代器，每个元素为：
                - Image: 孤立添加，只添加图像本身
                - (Image, PackSequenceIndex): 添加图像并绑定到序列
            batch_size: 批处理大小
        """

        def convert_image_item(image_item):
            duckdb_data = defaultdict(list)
            lance_images = []
            lance_annotations = []

            if isinstance(image_item, tuple):
                # 序列绑定模式: (Image, PackSequenceIndex)
                img, pack_index = image_item
                bind_to_sequence = True
                index = pack_index.index
                sequence_id = pack_index.sequence_id
            else:
                # 孤立添加模式: Image
                img = image_item
                bind_to_sequence = False
                index = None
                sequence_id = None

            # 处理图像
            lance_image = LanceImage.from_image(img)
            lance_images.append(lance_image)

            # 添加图像到 images 表
            duckdb_data["images"].append(
                dict(
                    id=img.id.to_uuid(),
                    uri=img.uri,
                    source=img.source,
                    width=lance_image.width,
                    height=lance_image.height,
                )
            )

            # 如果需要绑定到序列，添加到 sequence_images 表
            if bind_to_sequence and sequence_id is not None and index is not None:
                duckdb_data["sequence_images"].append(
                    dict(
                        sequence_id=sequence_id.to_uuid(),
                        image_id=img.id.to_uuid(),
                        index=self._as_string_index(index),
                    )
                )

            # 处理注释
            if img.annotations is not None:
                for anno in img.annotations:
                    annotation_id = self._process_annotation(
                        anno, duckdb_data, lance_annotations
                    )
                    duckdb_data["image_annotations"].append(
                        dict(
                            image_id=img.id.to_uuid(),
                            annotation_id=annotation_id,
                        )
                    )

            return dict(
                duckdb=duckdb_data,
                lance=dict(
                    images=LanceImage.to_batch(lance_images) if lance_images else None,
                    annotations=LanceAnnotation.to_batch(lance_annotations)
                    if lance_annotations
                    else None,
                ),
            )

        handler = duckdb_handler or self.meta_handler
        if handler is None:
            raise ValueError("duckdb_handler is required")

        lance_tables = {
            name: DistributedLanceWriter(self.lance_uris[name], schema=schema)
            for name, schema in self.LANCE_SCHEMA.items()
        }

        for items in batched(map(convert_image_item, images), batch_size=batch_size):
            batch = self._merge_batch(items, lance_names=list(self.LANCE_SCHEMA.keys()))

            try:
                for name, table in batch["lance"].items():
                    table: pa.Table
                    if table is not None and not table.num_rows == 0:
                        lance_tables[name].write_batch(table)

                self._batch_insert_duckdb(handler, batch["duckdb"])
            except Exception:
                logger.error(f"{traceback.format_exc()}")

    def add_sample_annotations(
        self,
        sample_annotations: Iterable[SampleAnnotation],
        batch_size: int = 100_000,
        duckdb_handler: DuckDBHandler | None = None,
    ):
        handler = duckdb_handler or self.meta_handler
        if handler is None:
            raise ValueError("duckdb_handler is required")

        def convert_sample_annotations_batch(sample_annotations_batch):
            duckdb_data = defaultdict(list)

            seen_creators = set()

            for sample_annotation in sample_annotations_batch:
                if sample_annotation.creator.id not in seen_creators:
                    seen_creators.add(sample_annotation.creator.id)
                    duckdb_data["creators"].append(
                        dict(
                            id=sample_annotation.creator.id.to_uuid(),
                            name=sample_annotation.creator.name,
                            meta=sample_annotation.creator.json_meta,
                        )
                    )

                if isinstance(sample_annotation.value, Number):
                    value_float = sample_annotation.value
                    value_json = None
                else:
                    value_float = None
                    value_json = jsonify_meta(sample_annotation.value)

                duckdb_data["sample_annotations"].append(
                    dict(
                        id=sample_annotation.id.to_uuid(),
                        name=sample_annotation.name,
                        creator_id=sample_annotation.creator.id.to_uuid(),
                        sequence_id=sample_annotation.sequence_id.to_uuid(),
                        value_float=value_float,
                        value_json=value_json,
                    )
                )

                for participant in sample_annotation.participants:
                    duckdb_data["sample_annotation_elements"].append(
                        dict(
                            sample_annotation_id=sample_annotation.id.to_uuid(),
                            element_id=participant[0].to_uuid(),
                            element_type=participant[1].value,
                            role=participant[2],
                        )
                    )

            return dict(duckdb=duckdb_data, lance=dict())

        for items_batch in batched(sample_annotations, batch_size):
            batch = convert_sample_annotations_batch(items_batch)

            try:
                self._batch_insert_duckdb(handler, batch["duckdb"])
            except Exception:
                logger.error(f"{traceback.format_exc()}")

    def add_annotations(
        self,
        annotations: Iterable[Annotation | tuple[Annotation, ID, str]],
        batch_size: int = 100_000,
        duckdb_handler: DuckDBHandler | None = None,
    ):
        """
        批量添加注释，支持孤立添加和关联绑定两种模式

        Args:
            annotations: 注释数据迭代器，每个元素为：
                - Annotation: 孤立添加，只添加注释本身
                - (Annotation, ID, str): 添加注释并绑定到 image 或 text
                    - ID: 关联的 image_id 或 text_id
                    - str: 元素类型，"image" 或 "text"
            batch_size: 批处理大小
            duckdb_handler: 可选的 DuckDB 处理器
        """
        handler = duckdb_handler or self.meta_handler
        if handler is None:
            raise ValueError("duckdb_handler is required")

        def convert_annotations_batch(annotations_batch):
            duckdb_data = defaultdict(list)
            lance_annotations = []

            seen_creators = set()

            for anno_item in annotations_batch:
                if isinstance(anno_item, tuple):
                    # 关联绑定模式: (Annotation, ID, str)
                    anno, element_id, element_type = anno_item
                    bind_to_element = True
                else:
                    # 孤立添加模式: Annotation
                    anno = anno_item
                    bind_to_element = False
                    element_id = None
                    element_type = None

                # 处理 creator
                creator_id = None
                if anno.creator is not None:
                    creator_id = anno.creator.id
                    if creator_id not in seen_creators:
                        seen_creators.add(creator_id)
                        duckdb_data["creators"].append(
                            dict(
                                id=creator_id,
                                name=anno.creator.name,
                                meta=anno.creator.json_meta,
                            )
                        )

                # 添加到 annotations 表
                duckdb_data["annotations"].append(
                    dict(
                        id=anno.id.to_uuid(),
                        name=anno.name,
                        type=anno.type_,
                        creator_id=creator_id,
                        meta=anno.json_meta,
                    )
                )

                # 如果需要绑定到元素，添加到关联表
                if bind_to_element and element_id is not None and element_type:
                    if element_type == "image":
                        duckdb_data["image_annotations"].append(
                            dict(
                                image_id=element_id.to_uuid(),
                                annotation_id=anno.id.to_uuid(),
                            )
                        )
                    elif element_type == "text":
                        duckdb_data["text_annotations"].append(
                            dict(
                                text_id=element_id.to_uuid(),
                                annotation_id=anno.id.to_uuid(),
                            )
                        )

                # 如果有 blob，添加到 Lance
                if anno.blob is not None:
                    lance_annotations.append(LanceAnnotation.from_annotation(anno))

            lance_data: dict = {}
            if lance_annotations:
                lance_data["annotations"] = LanceAnnotation.to_batch(lance_annotations)

            return dict(duckdb=duckdb_data, lance=lance_data)

        lance_tables = {
            "annotations": DistributedLanceWriter(
                self.lance_uris["annotations"], schema=self.LANCE_SCHEMA["annotations"]
            )
        }

        for items_batch in batched(annotations, batch_size):
            batch = convert_annotations_batch(items_batch)

            try:
                for name, table in batch["lance"].items():
                    table: pa.Table
                    if table is not None and not table.num_rows == 0:
                        lance_tables[name].write_batch(table)

                self._batch_insert_duckdb(handler, batch["duckdb"])
            except Exception:
                logger.error(f"{traceback.format_exc()}")

    def associate_images(
        self, items: Iterable[tuple[ID, PackSequenceIndex]], batch_size: int = 100_000
    ):
        """
        将已有的图像ID关联到序列中

        Args:
            items: 图像ID和序列索引的迭代器, 每个元素为 (image_id, PackSequenceIndex)
            batch_size: 批处理大小
        """
        handler = self.meta_handler

        def convert_associations_batch(associations_batch):
            duckdb_data = defaultdict(list)

            for image_id, pack_index in associations_batch:
                duckdb_data["sequence_images"].append(
                    dict(
                        sequence_id=pack_index.sequence_id.to_uuid(),
                        image_id=image_id.to_uuid(),
                        index=self._as_string_index(pack_index.index),
                    )
                )

            return dict(duckdb=duckdb_data, lance=dict())

        for items_batch in batched(items, batch_size):
            batch = convert_associations_batch(items_batch)

            try:
                self._batch_insert_duckdb(handler, batch["duckdb"])
            except Exception:
                logger.error(f"{traceback.format_exc()}")

    def associate_texts(
        self, items: Iterable[tuple[ID, PackSequenceIndex]], batch_size: int = 100_000
    ):
        """
        将已有的文本ID关联到序列中

        Args:
            items: 文本ID和序列索引的迭代器, 每个元素为 (text_id, PackSequenceIndex)
            batch_size: 批处理大小
        """
        handler = self.meta_handler

        def convert_associations_batch(associations_batch):
            duckdb_data = defaultdict(list)

            for text_id, pack_index in associations_batch:
                duckdb_data["sequence_texts"].append(
                    dict(
                        sequence_id=pack_index.sequence_id.to_uuid(),
                        text_id=text_id.to_uuid(),
                        index=self._as_string_index(pack_index.index),
                    )
                )

            return dict(duckdb=duckdb_data, lance=dict())

        for items_batch in batched(items, batch_size):
            batch = convert_associations_batch(items_batch)

            try:
                self._batch_insert_duckdb(handler, batch["duckdb"])
            except Exception:
                logger.error(f"{traceback.format_exc()}")

    def associate_annotations(
        self,
        items: Iterable[tuple[ID, ID, str]],
        batch_size: int = 100_000,
        duckdb_handler: DuckDBHandler | None = None,
    ):
        """
        将已有的注释ID关联到图像或文本

        Args:
            items: 关联项的迭代器, 每个元素为 (annotation_id, element_id, element_type)
                - annotation_id: 注释ID
                - element_id: 图像ID 或 文本ID
                - element_type: 元素类型，"image" 或 "text"
            batch_size: 批处理大小
            duckdb_handler: 可选的 DuckDB 处理器
        """
        handler = duckdb_handler or self.meta_handler
        if handler is None:
            raise ValueError("duckdb_handler is required")

        def convert_associations_batch(associations_batch):
            duckdb_data = defaultdict(list)

            for annotation_id, element_id, element_type in associations_batch:
                if element_type == "image":
                    duckdb_data["image_annotations"].append(
                        dict(
                            image_id=element_id.to_uuid(),
                            annotation_id=annotation_id.to_uuid(),
                        )
                    )
                elif element_type == "text":
                    duckdb_data["text_annotations"].append(
                        dict(
                            text_id=element_id.to_uuid(),
                            annotation_id=annotation_id.to_uuid(),
                        )
                    )

            return dict(duckdb=duckdb_data, lance=dict())

        for items_batch in batched(items, batch_size):
            batch = convert_associations_batch(items_batch)

            try:
                self._batch_insert_duckdb(handler, batch["duckdb"])
            except Exception:
                logger.error(f"{traceback.format_exc()}")

    def commit(self):
        DistributedDuckDBWriter(
            handler=self.meta_handler, table_names=self._DUCKDB_TABLE_NAME_ORDER
        ).commit()

        for name, schema in self.LANCE_SCHEMA.items():
            DistributedLanceWriter(self.lance_uris[name], schema=schema).commit()

    def merge(
        self,
        duckdb_files: list[str | Path] | None = None,
        lance_files: list[str | Path] | None = None,
        remove_original: bool = False,
    ):
        if duckdb_files is not None:
            DistributedDuckDBWriter(
                handler=self.meta_handler, table_names=self._DUCKDB_TABLE_NAME_ORDER
            ).merge(duckdb_files, remove_original=remove_original)

        if lance_files is not None:
            raise NotImplementedError("Lance files merging is not implemented")

    def get_sequence_ids_by_sources(self, sources: Sequence[str]) -> list[ID]:
        if isinstance(sources, str):
            sources = [sources]
        placeholders = ",".join(["?" for _ in sources])
        query = f"SELECT id FROM sequences WHERE source IN ({placeholders})"

        sequences = self.meta_handler.query_batch(query, sources)
        sequence_ids = [ID.from_(s["id"]) for s in sequences]
        return sequence_ids

    def get_sequence_metas(self, sequence_ids: Sequence[ID]) -> list[dict]:
        out = self.meta_handler.query_batch(
            load_duckdb_sql("sql/get_sequences.sql"),
            [tuple(sid.to_uuid() for sid in sequence_ids)],
        )
        return convert_as_id(out)

    def get_image_ids_by_uris(
        self, uris: Sequence[str], source: str | None = None
    ) -> dict[str, ID]:
        placeholders = ",".join(["?" for _ in uris])
        if source is None:
            image_ids = self.meta_handler.query_batch(
                f"SELECT id, uri, source FROM images WHERE uri IN ({placeholders})",
                tuple(uris),
            )
        else:
            image_ids = self.meta_handler.query_batch(
                f"SELECT id, uri, source FROM images WHERE uri IN ({placeholders}) AND source = ?",
                tuple(uris),
                source,
            )
        return {item["uri"]: ID.from_(item["id"]) for item in image_ids}

    def get_image_bytes_by_ids(self, ids: Sequence[ID]) -> dict[ID, bytes]:
        images_lance = self.lance_taker.lance_dataset(self.lance_uris["images"])
        table: pa.Table = self.lance_taker.by_ids(
            images_lance, list(ids), columns=["id", "image"]
        )
        images = {
            ID.from_(id_): image_data
            for id_, image_data in zip(
                table.column("id").to_pylist(), table.column("image").to_pylist()
            )
        }
        return images

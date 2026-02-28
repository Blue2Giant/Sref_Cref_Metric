import json
import os
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence, Union

import lance
import lance.dataset
import numpy as np
import pyarrow as pa
from lance import LanceOperation
from lance.fragment import DEFAULT_MAX_BYTES_PER_FILE, FragmentMetadata, write_fragments
from loguru import logger

from vault.schema import ID

if TYPE_CHECKING:
    from lance.types import ReaderLike

# 创建一个从 Python 类型到 PyArrow 类型的映射
# This map can be extended to support more types like int, float, etc.
_PYTHON_TO_PYARROW_TYPE_MAP = {
    str: pa.string(),
    bytes: pa.binary(),
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
}


@dataclass(frozen=True)
class LanceItem:
    @classmethod
    def get_schema(cls) -> pa.Schema:
        raise NotImplementedError()

    @classmethod
    def _to_pydict(cls, items: list["LanceItem"]) -> dict[str, list]:
        return {
            field.name: [getattr(item, field.name) for item in items]
            for field in fields(cls)
        }

    @classmethod
    def to_batch(cls, items: list["LanceItem"]) -> pa.RecordBatch:
        """转为 Arrow RecordBatch"""
        schema = cls.get_schema()
        if not items:
            # 每个字段一个空数组, 保证 schema 对齐
            arrays = [pa.array([], type=f.type) for f in schema]
            return pa.RecordBatch.from_arrays(arrays, schema=schema)
        return pa.RecordBatch.from_pydict(cls._to_pydict(items), schema=schema)

    @classmethod
    def to_table(cls, items: list["LanceItem"]) -> pa.Table:
        """转为 Arrow Table"""
        schema = cls.get_schema()
        if not items:
            return schema.empty_table()
        return pa.Table.from_pydict(cls._to_pydict(items), schema=schema)


class DistributedLanceWriter:
    """
    Manages distributed writes to a Lance dataset for concurrent environments.

    This class facilitates a two-phase commit process suitable for multi-process or
    multi-threaded applications.

    **Workflow:**

    1.  **Initialization**: Create an instance of this class in both the coordinator
        and worker processes.
    2.  **Phase 1 (Workers)**: Each worker process calls the `write_batch()` method
        to write its own data shard. This action creates a Lance data fragment and
        simultaneously saves its corresponding metadata to a unique, temporary file
        in a special cache directory (`_meta_cache`).
    3.  **Phase 2 (Coordinator)**: After all workers complete, a single coordinator
        process calls the `commit()` method. This method atomically reads all the
        cached metadata files, commits all the fragments to the dataset in a single
        operation, and then cleans up the cache.

    This design avoids race conditions by ensuring each worker writes to its own
    metadata file and that the final dataset modification is a single, atomic commit.
    """

    META_CACHE_DIR = "_lance_fragments/"

    def __init__(self, uri: Union[str, Path], schema: pa.Schema, mode: str = "append"):
        """
        Initializes the distributed writer.

        Args:
            uri (Union[str, Path]): The root URI of the Lance dataset.
            schema (pa.Schema): The PyArrow schema of the data.
            mode (str): The write mode. Must be 'append' or 'overwrite'.
                        - 'append': Adds data to an existing dataset. Creates the
                                    dataset if it does not exist.
                        - 'overwrite': Creates a new dataset, completely replacing
                                       any existing one at the same URI.
        """
        if mode not in ("append", "overwrite"):
            raise ValueError("Mode must be either 'append' or 'overwrite'.")

        self.uri = str(uri)
        self.schema = schema
        self.mode = mode
        self.meta_cache_path = Path(self.uri) / self.META_CACHE_DIR

        # A unique ID for this writer instance to prevent filename collisions,
        # especially important in a distributed environment.
        self.worker_id = f"{os.getpid()}_{uuid.uuid4().hex}"
        self.meta_file = self.meta_cache_path / f"meta_{self.worker_id}.jsonl"

        self._setup()

    def _setup(self):
        """Creates the metadata cache directory if it doesn't exist."""
        self.meta_cache_path.mkdir(parents=True, exist_ok=True)

    def write_batch(
        self,
        data: "ReaderLike",
        max_rows_per_file: int = 1024 * 1024,
        max_rows_per_group: int = 1024,
    ) -> int:
        """
        Writes a batch of data as a new fragment (Worker-side method).

        This method writes the data to a new Lance fragment within the dataset URI
        and records the fragment's metadata to a temporary cache file for the final commit.

        Args:
            data: The data to write. Can be a PyArrow Table, RecordBatch, or a
                  Python dictionary of lists.

        Returns:
            int: The number of fragments written in this batch.
        """

        fragments = write_fragments(
            data,
            self.uri,
            schema=self.schema,
            max_rows_per_file=max_rows_per_file,
            max_rows_per_group=max_rows_per_group,
            max_bytes_per_file=DEFAULT_MAX_BYTES_PER_FILE,
        )

        # Persist metadata to a unique file for the commit phase.
        # The 'a' (append) mode allows a single worker to call write_batch multiple times.
        with open(self.meta_file, "a") as f:
            for fragment in fragments:
                f.write(json.dumps(fragment.to_json()) + "\n")

        return len(fragments)

    def _collect_fragments(self) -> tuple[List[FragmentMetadata], List[Path]]:
        """
        扫描缓存目录, 批量收集和反序列化所有片段元数据。

        Returns:
            A tuple containing:
            - A list of all deserialized FragmentMetadata objects.
            - A list of paths to the metadata files that were processed.
        """
        all_fragments: List[FragmentMetadata] = []
        processed_files: List[Path] = []

        if not self.meta_cache_path.exists():
            return [], []

        for meta_file in self.meta_cache_path.glob("*.jsonl"):
            processed_files.append(meta_file)
            try:
                # 一次性读取整个文件内容，减少I/O操作
                with open(meta_file, "r") as f:
                    content = f.read()

                # 批量处理所有行
                for line in content.strip().split("\n"):
                    # 忽略空行
                    if line.strip():
                        all_fragments.append(FragmentMetadata.from_json(line))
            except (json.JSONDecodeError, IOError) as e:
                # If a metadata file is corrupt, it's safer to fail the whole commit
                # rather than risk a partial or incorrect dataset state.
                raise IOError(
                    f"Fatal: Failed to read or parse metadata from {meta_file}. "
                    f"Commit aborted. Please inspect the file. Error: {e}"
                )

        return all_fragments, processed_files

    def commit(self) -> None:
        """
        Commits all written fragments to the dataset (Coordinator-side method).

        This method must be called by a single process after all workers have
        finished their `write_batch` calls. It reads all cached metadata, commits the
        new fragments to the dataset, and cleans up the cache directory upon success.
        """
        all_fragments, processed_files = self._collect_fragments()

        if not all_fragments:
            # Cleanup only on successful commit.
            for path in processed_files:
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(
                        f"Coordinator: Warning - Failed to remove metadata file {path}: {e}"
                    )

            if self.meta_cache_path.exists() and not any(
                self.meta_cache_path.iterdir()
            ):
                os.rmdir(self.meta_cache_path)

            logger.info(
                f"No new fragments found in {self.meta_cache_path}. Nothing to commit."
            )
            return

        logger.info(
            f"Found {len(all_fragments)} fragments in {self.meta_cache_path} from "
            f"{len(processed_files)} workers to commit. "
        )

        try:
            if self.mode == "overwrite":
                op = LanceOperation.Overwrite(self.schema, all_fragments)
                # For a clean overwrite, we start from a non-existent version (version 0).
                # Lance's commit operation handles the removal of old data files.
                read_version = 0
            elif self.mode == "append":
                try:
                    dataset = lance.dataset(self.uri)
                    read_version = dataset.version
                    # Schema is not passed for append; it's inferred from the dataset.
                    op = LanceOperation.Append(all_fragments)
                except (FileNotFoundError, ValueError):
                    # This handles the case where the dataset doesn't exist yet.
                    # An 'append' to a non-existent dataset is effectively a 'create'.
                    logger.info(
                        f"Coordinator: Dataset not found at '{self.uri}'. "
                        f"Creating new dataset from fragments."
                    )
                    read_version = 0
                    op = LanceOperation.Overwrite(self.schema, all_fragments)
            else:
                # This case is already handled in __init__, but as a safeguard.
                raise RuntimeError(f"Internal error: unsupported mode '{self.mode}'")

            lance.LanceDataset.commit(
                self.uri,
                op,
                read_version=read_version,
            )
            logger.success(
                f"Coordinator: Successfully committed {len(all_fragments)} fragments."
            )

            # 检查并创建id列的索引
            self._create_id_index_if_exists()

        except Exception as e:
            # If the commit fails, we do NOT clear the cache. This allows for
            # manual inspection or a retry of the commit operation.
            logger.error(
                f"Coordinator: Error during commit: {e}. "
                f"Metadata cache will NOT be cleared to allow for retry."
            )
            raise
        else:
            # Cleanup only on successful commit.
            for path in processed_files:
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(
                        f"Coordinator: Warning - Failed to remove metadata file {path}: {e}"
                    )

            if not any(self.meta_cache_path.iterdir()):
                os.rmdir(self.meta_cache_path)

            logger.debug(
                f"Metadata cache in {self.meta_cache_path} cleared successfully."
            )

    def _create_id_index_if_exists(self) -> None:
        """
        检查schema中是否存在id列, 如果存在则为其创建BTREE索引。

        BTREE索引适合唯一标识符查询, 能够提供良好的查询性能。
        """
        try:
            # 打开数据集检查schema
            dataset = lance.dataset(self.uri)
            schema = dataset.schema

            # 检查是否存在id列
            id_field = None
            for field in schema:
                if field.name == "id":
                    id_field = field
                    break

            if id_field is None:
                logger.debug("No 'id' column found in schema, skipping index creation.")
                return

            # 为id列创建BTREE索引
            # BTREE索引适合唯一标识符, 提供良好的等值查询和范围查询性能
            dataset.create_scalar_index(
                column="id", index_type="BTREE", name="id_btree_idx", replace=True
            )

            logger.info(
                f"Successfully created BTREE index for 'id' column in dataset at '{self.uri}'"
            )

        except Exception as e:
            # 索引创建失败不应该影响整个commit过程
            logger.warning(
                f"Failed to create index for 'id' column: {e}. "
                f"Dataset commit was successful, but queries on 'id' column may be slower."
            )


class LanceTaker:
    """
    高效的 Lance 数据集数据读取器, 支持多种查询方式。

    该类提供了从 Lance 数据集中读取数据的统一接口, 支持通过索引、行ID、
    唯一标识符或查询条件来获取数据。内部维护数据集缓存以提高性能。

    主要特性:
    - 支持多种查询方式:索引、行ID、唯一标识符、查询条件
    - 自动缓存数据集对象, 避免重复加载
    - 支持 PyTorch 多进程环境
    - 支持列选择和结果排序
    - 批量处理多个引用

    使用示例:
        >>> taker = LanceTaker(verbose=True)
        >>> refs = [
        ...     LanceTaker.Ref(lance_path="/path/to/dataset", index=0),
        ...     LanceTaker.Ref(lance_path="/path/to/dataset", index=123),
        ... ]
        >>> table = taker(refs)

    Attributes:
        verbose (bool): 是否启用详细日志输出
        _lance_datasets (dict[str, lance.LanceDataset]): 数据集路径到数据集对象的映射
        _lance_datasets_rows (dict[str, int]): 数据集路径到行数的映射
    """

    @dataclass
    class Ref:
        """
        数据引用, 用于指定从 Lance 数据集中获取哪些数据。

        每个 Ref 对象代表一个数据获取请求, 包含数据集路径和查询条件。
        支持多种查询方式, 但每次查询整组 Ref 只能使用其中一种。

        Attributes:
            lance_path (str): Lance 数据集的路径
            index (int | None): 行索引, 用于按位置获取数据
            row_id (int | None): 行ID, 用于按行ID获取数据
            query (tuple[str, Any] | None): 查询条件, 格式为 (列名, 值列表)
            id (ID | None): 唯一标识符, 用于按ID获取数据
            columns (tuple[str, ...] | str | None): 要获取的列名, None表示获取所有列

        Note:
            - index, row_id, query, id 四个参数中只能指定一个
            - 如果指定了多个查询参数, 会抛出异常
            - columns 参数可以与任何查询参数组合使用
        """

        lance_path: str
        index: int | None = None
        row_id: int | None = None
        query: tuple[str, Any] | None = None
        id: ID | None = None
        columns: tuple[str, ...] | str | None = None

    def __init__(self, verbose=False) -> None:
        """
        初始化 LanceTaker 实例。

        Args:
            verbose (bool, optional): 是否启用详细日志输出。默认为 False。
                                   当设置为 True 时, 会在创建数据集时输出调试信息。

        Note:
            初始化时会创建空的数据集缓存字典, 数据集会在首次访问时懒加载。
        """
        self.verbose = verbose
        self._lance_datasets: dict[str, lance.LanceDataset] = {}
        self._lance_datasets_rows: dict[str, int] = {}

    @staticmethod
    def pytorch_worker_info() -> tuple[int, int]:
        """
        获取当前 PyTorch 工作进程信息。

        该方法会尝试从环境变量或 PyTorch DataLoader 中获取当前工作进程的
        ID 和总工作进程数。主要用于多进程环境下的日志记录。

        Returns:
            tuple[int, int]: (worker_id, num_workers) 元组
                - worker_id: 当前工作进程ID, 默认为0
                - num_workers: 总工作进程数, 默认为1

        Note:
            - 首先检查环境变量 WORKER 和 NUM_WORKERS
            - 如果环境变量不存在, 尝试从 PyTorch DataLoader 获取
            - 如果 PyTorch 不可用, 返回默认值 (0, 1)
        """
        worker = 0
        num_workers = 1
        if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
            worker = int(os.environ["WORKER"])
            num_workers = int(os.environ["NUM_WORKERS"])
        else:
            try:
                import torch.utils.data  # type: ignore

                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    worker = worker_info.id
                    num_workers = worker_info.num_workers
            except ModuleNotFoundError:
                pass

        return worker, num_workers

    def lance_dataset(self, dataset_path: str | Path) -> lance.LanceDataset:
        """
        获取或创建 Lance 数据集对象。

        该方法实现了数据集的懒加载和缓存机制。首次访问某个数据集路径时,
        会创建 LanceDataset 对象并缓存, 后续访问直接返回缓存的对象。

        Args:
            dataset_path (str | Path): 数据集路径, 可以是字符串或 Path 对象

        Returns:
            lance.LanceDataset: Lance 数据集对象

        Note:
            - 数据集路径会被转换为绝对路径进行缓存
            - 首次加载时会计算并缓存数据集的行数
            - 如果启用了 verbose 模式, 会输出加载信息

        Raises:
            FileNotFoundError: 如果数据集路径不存在
            ValueError: 如果数据集格式不正确
        """
        dataset_path = Path(dataset_path).absolute().as_posix()

        if dataset_path not in self._lance_datasets:
            self._lance_datasets[dataset_path] = lance.dataset(dataset_path)
            self._lance_datasets_rows[dataset_path] = self._lance_datasets[
                dataset_path
            ].count_rows()

            if self.verbose:
                worker_id, num_worker = self.pytorch_worker_info()
                logger.debug(
                    (
                        f"[{worker_id}/{num_worker}] create lance.LanceDataset: {dataset_path}, "
                        f"found {self._lance_datasets_rows[dataset_path]} samples."
                    )
                )

        return self._lance_datasets[dataset_path]

    @staticmethod
    def in_or_equal(conditions: list[Any]) -> str:
        """
        将条件列表转换为 SQL 风格的过滤表达式。

        根据条件数量自动选择使用等号 (=) 或 IN 操作符。
        单个条件使用等号, 多个条件使用 IN 操作符。

        Args:
            conditions (list[Any]): 条件值列表

        Returns:
            str: SQL 风格的过滤表达式

        Examples:
            >>> LanceTaker.in_or_equal([1])
            '= 1'
            >>> LanceTaker.in_or_equal([1, 2, 3])
            "IN (1, 2, 3)"
            >>> LanceTaker.in_or_equal(['a', 'b'])
            "IN ('a', 'b')"

        Note:
            - 字符串值会自动添加单引号, 除非以 "X'" 开头（表示二进制数据)
            - 其他类型的值直接转换为字符串
        """

        def format_value(value):
            if isinstance(value, str) and not value.startswith("X'"):
                return f"'{value}'"
            return str(value)

        if len(conditions) == 1:
            return f"= {format_value(conditions[0])}"
        else:
            return f"IN ({', '.join(map(format_value, conditions))})"

    @staticmethod
    def by_indices(
        lance_dataset: lance.LanceDataset | str | Path,
        indices: list[int],
        columns: list[str] | None = None,
        sort_indices: bool = True,
        keep_order: bool = False,
    ) -> pa.Table:
        """
        通过行索引从数据集中获取数据。

        该方法使用 Lance 的 take 操作来获取指定索引位置的数据行。
        为了提高性能, 默认会对索引进行排序, 但可以选择保持原始顺序。

        Args:
            lance_dataset (lance.LanceDataset): Lance 数据集对象
            indices (list[int]): 要获取的行索引列表
            columns (list[str] | None, optional): 要获取的列名列表, None表示获取所有列
            sort_indices (bool, optional): 是否对索引进行排序以提高性能。默认为 True
            keep_order (bool, optional): 是否保持原始索引顺序。默认为 False

        Returns:
            pa.Table: 包含指定行数据的 PyArrow 表

        Note:
            - 当 sort_indices=True 时, 会先对索引排序, 然后重新排列结果以保持原始顺序
            - 当 keep_order=False 时, 返回的数据顺序可能与输入索引顺序不同
            - 索引超出范围会导致错误

        Raises:
            IndexError: 如果索引超出数据集范围
        """
        if isinstance(lance_dataset, str) or isinstance(lance_dataset, Path):
            lance_dataset = lance.dataset(lance_dataset)

        if sort_indices:
            # 使用 numpy 直接操作，避免中间列表和冗余数组操作
            indices_array = np.array(indices)
            sorted_idx = np.argsort(indices_array)

            # 获取排序后的索引进行查询
            table = lance_dataset.take(
                indices_array[sorted_idx].tolist(), columns=columns
            )

            if keep_order:
                # 直接使用 argsort 的逆操作恢复原始顺序
                inverse_idx = np.empty_like(sorted_idx)
                inverse_idx[sorted_idx] = np.arange(len(sorted_idx))
                table = table.take(inverse_idx)
        else:
            table = lance_dataset.take(indices, columns=columns)

        return table

    @staticmethod
    def by_row_ids(
        lance_dataset: lance.LanceDataset | str | Path,
        row_ids: list[int],
        columns: list[str] | None = None,
        keep_order: bool = False,
        **kwargs,
    ) -> pa.Table:
        """
        通过行ID从数据集中获取数据。

        该方法使用 Lance 的扫描器功能, 通过 _rowid 列来过滤数据。
        行ID是 Lance 数据集的内部行标识符, 与行索引不同。

        Args:
            lance_dataset (lance.LanceDataset | str | Path): Lance 数据集对象或数据集路径
            row_ids (list[int]): 要获取的行ID列表
            columns (list[str] | None, optional): 要获取的列名列表, None表示获取所有列
            keep_order (bool, optional): 是否保持原始行ID顺序。默认为 False

        Returns:
            pa.Table: 包含指定行数据的 PyArrow 表

        Note:
            - 如果传入字符串或Path, 会自动转换为 LanceDataset 对象
            - 行ID与行索引不同, 行ID是 Lance 的内部标识符
            - 当 keep_order=True 时, 会重新排列结果以匹配输入的行ID顺序
            - 不存在的行ID会被忽略, 不会导致错误
        """
        if isinstance(lance_dataset, str) or isinstance(lance_dataset, Path):
            lance_dataset = lance.dataset(lance_dataset)

        # 如果查询的列包含 image, 则设置 late_materialization 为 ["image"]
        if (
            columns is not None and "image" in columns
        ) and "late_materialization" not in kwargs:
            kwargs["late_materialization"] = ["image"]

        table = lance_dataset.scanner(
            filter=f"_rowid {LanceTaker.in_or_equal(row_ids)}",
            columns=columns,
            with_row_id=True,
            **kwargs,
        ).to_table()
        if not keep_order:
            return table

        row_id_to_original_index = {
            row_id.as_py(): i for i, row_id in enumerate(table.column("_rowid"))
        }
        take_indices = [row_id_to_original_index[rid] for rid in row_ids]
        return table.take(take_indices)

    @staticmethod
    def check_if_exist_index(
        lance_dataset: lance.LanceDataset | str | Path, field_name: str
    ) -> bool:
        """
        检查数据集中是否存在指定名称的索引。
        """
        if isinstance(lance_dataset, str) or isinstance(lance_dataset, Path):
            lance_dataset = lance.dataset(lance_dataset)
        for index in lance_dataset.list_indices():
            if isinstance(index, dict):
                fields = index["fields"]
            else:
                fields = index.fields
            if field_name in fields:
                return True
        return False

    @staticmethod
    def exist_id_index(lance_dataset: lance.LanceDataset | str | Path):
        return LanceTaker.check_if_exist_index(lance_dataset, "id")

    @staticmethod
    def by_query(
        lance_dataset: lance.LanceDataset | str | Path,
        key: str,
        values: list[Any],
        columns: list[str] | None = None,
        keep_order: bool = False,
        **kwargs,
    ) -> pa.Table:
        """
        通过查询条件从数据集中获取数据。

        该方法使用 Lance 的扫描器功能, 通过指定列的值来过滤数据。
        支持等值查询和 IN 查询, 根据值的数量自动选择查询方式。

        Args:
            lance_dataset (lance.LanceDataset | str | Path): Lance 数据集对象或数据集路径
            key (str): 要查询的列名
            values (list[Any]): 要匹配的值列表
            columns (list[str] | None, optional): 要获取的列名列表, None表示获取所有列
            keep_order (bool, optional): 是否保持原始值顺序。默认为 False

        Returns:
            pa.Table: 包含匹配行数据的 PyArrow 表

        Note:
            - 如果传入字符串或Path, 会自动转换为 LanceDataset 对象
            - 如果指定的列不在 columns 中, 会自动添加到查询列中
            - 当 keep_order=True 时, 会发出性能警告, 因为需要额外的排序操作
            - 排序基于值的字符串表示, 可能影响准确性
            - 不匹配的值会被忽略, 不会导致错误

        Warning:
            当 keep_order=True 时, 会发出 UserWarning 警告, 提醒用户性能和准确性限制。
        """
        if isinstance(lance_dataset, str) or isinstance(lance_dataset, Path):
            lance_dataset = lance.dataset(lance_dataset)
        if columns is not None and key not in columns:
            columns = [key] + (list(columns) or [])

        # 如果查询的列包含 image, 则设置 late_materialization 为 ["image"]
        if (
            columns is not None and "image" in columns
        ) and "late_materialization" not in kwargs:
            kwargs["late_materialization"] = ["image"]

        scanner = lance_dataset.scanner(
            filter=f"{key} {LanceTaker.in_or_equal(values)}",
            columns=columns,
            with_row_id=True,
            **kwargs,
        )
        table = scanner.to_table()

        if not keep_order:
            return table

        # 警告用户关于 keep_order 的性能和准确性限制
        import warnings

        warnings.warn(
            "使用 keep_order=True 可能有性能和准确性限制:\n"
            "1. 需要将值转换为字符串进行排序, 有性能消耗\n"
            "2. 不同对象可能产生相同的字符串表示, 影响排序准确性\n"
            "3. 建议使用 keep_order=False 获取结果后, 自己定义合理的排序策略",
            UserWarning,
            stacklevel=2,
        )

        # 创建 value 到原始顺序索引的映射 (优化版本: 一次性创建字典)
        value_to_index = {str(value): i for i, value in enumerate(values)}

        # 获取查询结果中的 key 列
        key_column = table.column(key)

        # 使用 numpy 加速排序索引生成 (单次遍历)
        sort_keys = np.array(
            [
                value_to_index.get(str(key_column[i].as_py()), len(values))
                for i in range(len(key_column))
            ]
        )

        # 使用 numpy 的 argsort 进行高效排序
        take_indices = np.argsort(sort_keys)

        return table.take(take_indices)

    @staticmethod
    def by_ids(
        lance_dataset: lance.LanceDataset | str | Path,
        ids: list[ID],
        columns: list[str] | None = None,
        keep_order: bool = False,
        **kwargs,
    ) -> pa.Table:
        """
        通过唯一标识符从数据集中获取数据。

        该方法专门用于通过 ID 列查询数据。ID 会被转换为十六进制字符串格式,
        并使用 X'...' 语法进行查询, 这是 Lance 中处理二进制数据的标准方式。

        Args:
            lance_dataset (lance.LanceDataset | str | Path): Lance 数据集对象或数据集路径
            ids (list[ID]): 要查询的ID列表
            columns (list[str] | None, optional): 要获取的列名列表, None表示获取所有列
            keep_order (bool, optional): 是否保持原始ID顺序。默认为 False

        Returns:
            pa.Table: 包含匹配行数据的 PyArrow 表

        Note:
            - 如果传入字符串或Path, 会自动转换为 LanceDataset 对象
            - ID 会被转换为 X'hex_string' 格式进行查询
            - 当 keep_order=True 时, 会重新排列结果以匹配输入的ID顺序
            - 不存在的ID会被忽略, 不会导致错误
            - 该方法内部调用了 by_query 方法

        """
        table = LanceTaker.by_query(
            lance_dataset=lance_dataset,
            key="id",
            values=[f"X'{id_.to_bytes().hex()}'" for id_ in ids],
            columns=columns,
            **kwargs,
        )
        if not keep_order:
            return table

        ids_to_index = {id_.as_py(): i for i, id_ in enumerate(table.column("id"))}

        take_indices = [ids_to_index[id_.to_uuid()] for id_ in ids]
        return table.take(take_indices)

    @staticmethod
    def take(
        lance_dataset: lance.LanceDataset | str | Path,
        columns: list[str] | None = None,
        row_ids: list[int] | None = None,
        indices: list[int] | None = None,
        ids: list[ID] | None = None,
        query: tuple[str, list[Any]] | None = None,
        keep_order: bool = False,
        **kwargs,
    ) -> pa.Table:
        """
        从数据集中获取数据的统一接口。

        该方法根据提供的参数自动选择合适的查询方式。支持多种查询参数,
        但每次调用只能使用其中一种查询方式。

        Args:
            lance_dataset (lance.LanceDataset | str | Path): Lance 数据集对象或数据集路径
            columns (list[str] | None, optional): 要获取的列名列表, None表示获取所有列
            row_ids (list[int] | None, optional): 行ID列表, 用于按行ID查询
            indices (list[int] | None, optional): 行索引列表, 用于按位置查询
            ids (list[ID] | None, optional): ID列表, 用于按唯一标识符查询
            query (tuple[str, list[Any]] | None, optional): 查询条件, 格式为 (列名, 值列表)
            keep_order (bool, optional): 是否保持原始顺序。默认为 False

        Returns:
            pa.Table: 包含查询结果的 PyArrow 表

        Note:
            - 如果传入字符串或Path, 会自动转换为 LanceDataset 对象
            - 查询参数优先级:row_ids > indices > ids > query
            - 如果提供了多个查询参数, 只会使用优先级最高的那个
            - columns 参数可以与任何查询参数组合使用

        Raises:
            ValueError: 如果没有提供任何查询参数
        """
        if isinstance(lance_dataset, str) or isinstance(lance_dataset, Path):
            lance_dataset = lance.dataset(lance_dataset)
        if row_ids is not None:
            return LanceTaker.by_row_ids(
                lance_dataset, row_ids, columns, keep_order, **kwargs
            )
        elif indices is not None:
            return LanceTaker.by_indices(
                lance_dataset, indices, columns, keep_order, **kwargs
            )
        elif ids is not None:
            return LanceTaker.by_ids(lance_dataset, ids, columns, keep_order, **kwargs)
        elif query is not None:
            key, values = query
            return LanceTaker.by_query(
                lance_dataset, key, values, columns, keep_order, **kwargs
            )
        else:
            raise ValueError("需要指定一个查询参数")

    def __call__(self, refs: Sequence["LanceTaker.Ref"]) -> pa.Table:
        """
        批量处理多个数据引用, 返回合并后的数据表。

        这是 LanceTaker 的主要接口方法。它接受一个引用列表,
        按数据集路径分组, 然后批量查询数据, 最后合并所有结果。

        Args:
            refs (Sequence[LanceTaker.Ref]): 数据引用列表

        Returns:
            pa.Table: 包含所有查询结果的合并 PyArrow 表

        Note:
            - 空引用列表会返回空表
            - 所有引用必须使用相同的列选择(columns 参数)
            - 引用会按数据集路径自动分组, 相同路径的引用会批量处理
            - 每个组内的引用必须使用相同的查询方式(索引、行ID、ID或查询)
            - 结果表是多个数据集查询结果的合并

        Raises:
            AssertionError: 如果引用使用了不同的列选择或查询方式
            ValueError: 如果引用没有指定有效的查询参数

        Examples:
            >>> taker = LanceTaker()
            >>> refs = [
            ...     LanceTaker.Ref(lance_path="/path/to/dataset", index=0),
            ...     LanceTaker.Ref(lance_path="/path/to/dataset", index=1),
            ... ]
            >>> table = taker(refs)
        """
        if not refs:
            # 返回空表
            return pa.table({})

        _columns = [r.columns for r in refs]
        # 将list转换为tuple以便可以放入set
        _columns_tuples = [tuple(col) if col is not None else None for col in _columns]
        assert len(set(_columns_tuples)) == 1, (
            f"一组样本只能有同一个columns, 但{set(_columns_tuples)=}"
        )
        columns = (_columns[0],) if isinstance(_columns[0], str) else _columns[0]

        lance_path_to_images: dict[str, list["LanceTaker.Ref"]] = dict()

        for r in refs:
            if r.lance_path not in lance_path_to_images:
                lance_path_to_images[r.lance_path] = [r]
            else:
                lance_path_to_images[r.lance_path].append(r)

        tables = []
        for path, _refs in lance_path_to_images.items():
            params = dict()
            if _refs[0].row_id is not None:
                assert all(r.row_id is not None for r in _refs)
                params["row_ids"] = [r.row_id for r in _refs]

            elif _refs[0].index is not None:
                assert all(r.index is not None for r in _refs)
                params["indices"] = [r.index for r in _refs]

            elif _refs[0].id is not None:
                assert all(r.id is not None for r in _refs)
                params["ids"] = [r.id for r in _refs]

            elif _refs[0].query is not None:
                assert all(r.query is not None for r in _refs)
                assert (
                    len(set(r.query[0] for r in _refs if r.query is not None)) == 1
                ), (
                    f"一组样本只能有同一个query, 但{set(r.query[0] for r in _refs if r.query is not None)=}"
                )
                params["query"] = (
                    _refs[0].query[0],
                    [r.query[1] for r in _refs if r.query is not None],
                )
            else:
                raise ValueError("需要指定一个查询参数")

            table: pa.Table = self.take(
                lance_dataset=self.lance_dataset(path),
                columns=list(columns) if columns is not None else None,
                **params,
                keep_order=False,
            )
            tables.append(table)

        return pa.concat_tables(tables) if len(tables) > 1 else tables[0]

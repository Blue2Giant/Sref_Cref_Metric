import concurrent.futures
import os
import re
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Iterable, List, Literal

import duckdb
import pandas as pd
import xxhash
from loguru import logger

MergeStrategy = Literal["INSERT OR IGNORE", "INSERT OR REPLACE"]


class DuckDBHandler:
    def __init__(
        self,
        schema: str,
        db_path: str | Path,
        read_only: bool = False,
    ):
        self.schema = schema
        self.db_path = str(db_path)
        self.read_only = read_only

        self._conn = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(database=self.db_path, read_only=self.read_only)

        assert self._conn is not None, "Database connection not initialized"
        return self._conn

    def create(self):
        assert not self.read_only
        db_path = Path(self.db_path)
        if not db_path.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn.execute(self.schema)

    def _dataframe_to_dataclass(self, df: pd.DataFrame, dc: type) -> list:
        if df.empty:
            return []
        records = df.to_dict("records")
        return [dc(**record) for record in records]  # type: ignore

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def add(self, data: list[dict], sql: str, dedup_by_id: bool = True):
        if not len(data):
            return

        df = pd.DataFrame(data)

        if dedup_by_id and "id" in df.columns:
            df = df.drop_duplicates(subset="id", keep="first")

        con = self.conn

        try:
            con.begin()
            con.register("df", df)
            con.execute(sql)
            con.commit()
            con.unregister("df")
        except Exception as e:
            con.rollback()
            logger.error(f"发生错误，事务已回滚: {e}")
            logger.error(f"{traceback.format_exc()}")

    def add_multiply(
        self,
        multiply_data: dict,
        sqls: list[str],
        table_names,
        dedup_by_id: bool = True,
    ):
        con = self.conn

        try:
            con.begin()

            for table_name, sql in zip(table_names, sqls):
                data = multiply_data[table_name]
                if not data:
                    continue
                df = pd.DataFrame(data)
                if dedup_by_id and "id" in df.columns:
                    df = df.drop_duplicates(subset="id", keep="first")
                con.register("df", df)
                con.execute(sql)
                con.unregister("df")

            con.commit()
        except Exception as e:
            con.rollback()
            logger.error(f"发生错误，事务已回滚: {e}")
            logger.error(f"{traceback.format_exc()}")

    def query_batch(self, query_sql: str, *args):
        con = self.conn

        # 使用参数化查询来防止SQL注入，DuckDB可以处理元组/列表作为IN子句的参数
        result = con.execute(query_sql, *args).fetchall()

        # 将数据库返回的、依赖于顺序的元组，动态地转换成自描述的、不依赖于顺序的字典
        output = []
        # con.description 是 DuckDB (以及遵循 Python DB-API 规范的其他数据库库)游标对象的一个属性
        # 在一个查询被执行之后，这个属性会包含结果集中每一列的描述信息。类似于
        # [
        #     ('sequence_id', 'UUID', None, None, None, None, None),
        #     ('images', 'LIST(STRUCT(id UUID, ...))', None, None, None, None, None),
        #     ('texts', 'LIST(STRUCT(id UUID, ...))', None, None, None, None, None)
        # ]
        assert con.description is not None, f"{query_sql=} is not a valid SQL query"
        column_names = [desc[0] for desc in con.description]

        for row in result:
            row_dict = dict(zip(column_names, row))
            output.append(row_dict)

        return output

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _merge_into(
    source_file: str,
    target_file: str,
    table_names: List[str],
    strategy: MergeStrategy,
) -> None:
    """
    把单个源文件合并到目标文件。失败时抛出异常。

    注意：此函数假设没有其他进程同时写入 target_file。
    """
    con = duckdb.connect(target_file, read_only=False)
    alias = f"source_{xxhash.xxh32_hexdigest(source_file)}"

    try:
        con.begin()
        con.execute(f"ATTACH '{source_file}' AS {alias} (READ_ONLY)")

        # 检查源文件中存在哪些表
        tables_sql = f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{alias}'"
        existing = {row[0] for row in con.execute(tables_sql).fetchall()}

        missing = [t for t in table_names if t not in existing]
        if missing:
            logger.warning(f"源文件 {source_file} 缺少表: {', '.join(missing)}")

        # 只合并存在的表
        for table in table_names:
            if table in existing:
                con.execute(f"{strategy} INTO {table} SELECT * FROM {alias}.{table}")

        con.execute(f"DETACH {alias}")
        con.commit()
        logger.debug(f"已合并: {source_file} -> {target_file}")

    except Exception as e:
        try:
            con.rollback()
        except Exception:
            pass
        raise RuntimeError(f"合并失败 {source_file} -> {target_file}: {e}") from e

    finally:
        con.close()


def _merge_group(
    group_files: List[str],
    staging_file: Path,
    table_names: List[str],
    strategy: MergeStrategy,
) -> Path:
    """
    处理单个分组：把一组文件串行合并到 staging 文件。

    流程：copy 第一个文件为 staging，然后串行 merge 其余文件。
    返回 staging 文件路径。失败时抛出异常。
    """
    if not group_files:
        raise ValueError("group_files 不能为空")

    # 复制第一个文件作为 staging
    shutil.copy2(group_files[0], staging_file)
    logger.debug(f"组 staging 初始化: {group_files[0]} -> {staging_file}")

    # 串行合并其余文件
    for source in group_files[1:]:
        _merge_into(source, str(staging_file), table_names, strategy)

    return staging_file


def _partition(items: List[str], n: int) -> List[List[str]]:
    """把列表平均分成 n 份。"""
    if n <= 0:
        raise ValueError("n must be positive")
    if not items:
        return []

    k, remainder = divmod(len(items), n)
    groups = []
    start = 0
    for i in range(n):
        size = k + (1 if i < remainder else 0)
        if size > 0:
            groups.append(items[start : start + size])
            start += size
    return groups


def grouped_merge(
    target_file: str,
    schema: str,
    table_names: List[str],
    source_files: List[str],
    strategy: MergeStrategy = "INSERT OR IGNORE",
    max_workers: int = 4,
    remove_on_success: bool = False,
) -> List[str]:
    """
    分组并行合并：组内串行，组间并行，最后串行写入目标。

    架构:
        源文件分成 N 组 -> 每组并行合并到各自 staging -> staging 串行合并到 target

    保证:
        任何时刻都不存在并发写入同一个文件的情况。

    Args:
        target_file: 最终目标数据库路径
        schema: 数据库 schema（用于创建目标文件）
        table_names: 要合并的表名列表
        source_files: 源文件列表
        strategy: 合并策略
        max_workers: 并行组数
        remove_on_success: 成功后是否删除源文件

    Returns:
        成功合并的源文件列表

    Raises:
        RuntimeError: 合并失败时抛出，staging 文件保留供检查
    """
    if not source_files:
        logger.warning("没有源文件需要合并")
        return []

    target_path = Path(target_file)
    staging_dir = target_path.parent / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 分组
    groups = _partition(source_files, max_workers)
    logger.info(f"开始分组合并: {len(source_files)} 文件 -> {len(groups)} 组 -> {target_file}")

    staging_files: List[Path] = []

    try:
        # 阶段1: 组间并行，组内串行
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {}
            for i, group in enumerate(groups):
                staging = staging_dir / f"staging_{i}_{uuid.uuid4().hex[:8]}.duckdb"
                future = executor.submit(
                    _merge_group, group, staging, table_names, strategy
                )
                futures[future] = (i, staging)

            for future in concurrent.futures.as_completed(futures):
                i, staging = futures[future]
                try:
                    result = future.result()
                    staging_files.append(result)
                    logger.debug(f"组 {i} 完成: {result}")
                except Exception as e:
                    logger.error(f"组 {i} 失败: {e}")
                    raise RuntimeError(f"分组 {i} 合并失败，staging 目录保留: {staging_dir}") from e

        # 阶段2: 串行合并到目标
        logger.info(f"所有分组完成，开始串行合并 {len(staging_files)} 个 staging 到目标")

        # 确保目标文件存在
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(target_path)) as con:
                con.execute(schema)
            logger.info(f"已创建目标文件: {target_file}")

        for staging in staging_files:
            _merge_into(str(staging), target_file, table_names, strategy)

        logger.info("合并完成!")

        # 清理 staging 文件
        for staging in staging_files:
            staging.unlink(missing_ok=True)
        if staging_dir.exists() and not any(staging_dir.iterdir()):
            staging_dir.rmdir()

        # 删除源文件
        if remove_on_success:
            logger.info(f"删除 {len(source_files)} 个源文件...")
            for f in source_files:
                Path(f).unlink(missing_ok=True)

        return source_files

    except Exception:
        logger.error(f"合并失败，staging 目录保留: {staging_dir}")
        raise


class DistributedDuckDBWriter:
    META_CACHE_DIR = "_duckdb"

    def __init__(
        self,
        handler: DuckDBHandler,
        table_names: list[str] | None = None,
        replace_on_merge: bool = False,
    ):
        self.final_db_path = Path(handler.db_path)
        self.schema = handler.schema

        # 从 schema 中解析出所有表名，用于后续合并操作
        self.table_names = table_names or self._parse_table_names(handler.schema)
        if not self.table_names:
            raise ValueError(
                "Could not parse any table names from the provided schema."
            )

        # 创建一个唯一的临时目录来存放所有子进程的 DB 文件
        self.temp_dir = self.final_db_path.parent / self.META_CACHE_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.worker_db_path = self.temp_dir / f"{os.getpid()}_{uuid.uuid4().hex}.duckdb"

        # 默认使用 INSERT OR IGNORE 以避免并发合并时的事务冲突
        self.merge_strategy: MergeStrategy = (
            "INSERT OR REPLACE" if replace_on_merge else "INSERT OR IGNORE"
        )

    @staticmethod
    def _parse_table_names(schema: str) -> List[str]:
        """从 CREATE TABLE 语句中解析表名。"""
        # 正则表达式，用于匹配 'CREATE TABLE table_name' 或 'CREATE TABLE IF NOT EXISTS table_name'
        return re.findall(
            r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?(\w+)", schema, re.IGNORECASE
        )

    def get_worker_handler(self) -> DuckDBHandler:
        handler = DuckDBHandler(self.schema, self.worker_db_path)
        handler.create()
        return handler

    def merge(
        self,
        temp_db_files: Iterable[str | Path],
        remove_original: bool = False,
        max_workers: int = 4,
    ):
        """合并指定的临时数据库文件到最终数据库。"""
        grouped_merge(
            target_file=str(self.final_db_path),
            schema=self.schema,
            table_names=self.table_names,
            source_files=[str(p) for p in temp_db_files],
            strategy=self.merge_strategy,
            max_workers=max_workers,
            remove_on_success=remove_original,
        )

    def commit(self, remove_original: bool = True, max_workers: int = 4):
        """
        将所有临时的 worker 数据库通过分组并行合并写入最终数据库。

        流程：
        1. 把临时文件分成 N 组，组内串行合并到各自 staging
        2. 组间并行执行
        3. 把所有 staging 串行合并到最终目标

        Args:
            remove_original: 是否删除原始 worker 文件
            max_workers: 并行组数

        Raises:
            RuntimeError: 合并失败时抛出
        """
        temp_db_files = [str(p) for p in self.temp_dir.glob("*.duckdb")]

        if not temp_db_files:
            logger.warning("No temporary DB files found to merge.")
            self._cleanup_temp_dir()
            return

        logger.info(f"Merging {len(temp_db_files)} temporary DB files into final DB...")
        grouped_merge(
            target_file=str(self.final_db_path),
            schema=self.schema,
            table_names=self.table_names,
            source_files=temp_db_files,
            strategy=self.merge_strategy,
            max_workers=max_workers,
            remove_on_success=remove_original,
        )
        self._cleanup_temp_dir()
        logger.info("Distributed commit completed.")

    def _cleanup_temp_dir(self):
        """安全地删除临时目录。"""
        try:
            if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Temporary directory '{self.temp_dir}' has been removed.")
        except OSError as e:
            logger.error(f"Error removing temporary directory '{self.temp_dir}': {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # 如果 `with` 块中出现异常，打印日志但不进行合并
            # 因为此时子进程可能已异常退出，临时文件状态未知
            logger.error(
                "An exception occurred within the 'with' block. Skipping merge."
            )
            logger.error(f"Exception Type: {exc_type}, Value: {exc_val}")
            logger.error("".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
            logger.warning(
                "Temporary files may be left in '%s' for inspection.", self.temp_dir
            )
        else:
            # 如果 `with` 块正常退出，执行合并和清理
            self.commit()

import os
from pathlib import Path
from unittest.mock import patch

import lance
import pyarrow as pa
import pytest

from vault.backend.lance import LanceTaker
from vault.schema import ID

# ==============================================================================
#  Test Fixtures
# ==============================================================================


@pytest.fixture
def temp_lance_dataset(tmp_path: Path) -> str:
    """创建一个临时的Lance数据集用于测试"""
    dataset_path = str(tmp_path / "test_dataset.lance")

    # 创建测试数据
    data = {
        "id": [ID.random().to_bytes() for _ in range(10)],
        "name": [f"item_{i}" for i in range(10)],
        "value": list(range(10)),
        "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "score": [0.1 * i for i in range(10)],
    }

    # 创建Arrow表并写入Lance数据集
    table = pa.table(data)
    lance.write_dataset(table, dataset_path)

    return dataset_path


@pytest.fixture
def temp_lance_dataset_with_ids(tmp_path: Path) -> tuple[str, list[ID]]:
    """创建一个包含已知ID的Lance数据集"""
    dataset_path = str(tmp_path / "test_dataset_with_ids.lance")

    # 创建已知的ID列表
    test_ids = [ID.random() for _ in range(5)]

    data = {
        "id": [id_.to_bytes() for id_ in test_ids],
        "name": [f"item_{i}" for i in range(5)],
        "value": list(range(5)),
    }

    table = pa.table(data)
    lance.write_dataset(table, dataset_path)

    return dataset_path, test_ids


@pytest.fixture
def lance_taker():
    """创建LanceTaker实例"""
    return LanceTaker(verbose=False)


# ==============================================================================
#  Test Static Methods
# ==============================================================================


def test_in_or_equal():
    """测试in_or_equal静态方法"""
    # 单个条件
    assert LanceTaker.in_or_equal([1]) == "= 1"
    assert LanceTaker.in_or_equal(["test"]) == "= 'test'"

    # 多个条件
    assert LanceTaker.in_or_equal([1, 2, 3]) == "IN (1, 2, 3)"
    assert LanceTaker.in_or_equal(["a", "b", "c"]) == "IN ('a', 'b', 'c')"


def test_by_indices(lance_taker, temp_lance_dataset):
    """测试by_indices静态方法"""
    dataset = lance.dataset(temp_lance_dataset)

    # 测试基本功能
    indices = [0, 2, 4]
    table = LanceTaker.by_indices(dataset, indices)
    assert len(table) == 3
    assert table.column("value").to_pylist() == [0, 2, 4]

    # 测试指定列
    table = LanceTaker.by_indices(dataset, indices, columns=["name", "value"])
    assert table.column_names == ["name", "value"]
    assert table.column("value").to_pylist() == [0, 2, 4]

    # 测试保持顺序
    indices = [4, 0, 2]
    table = LanceTaker.by_indices(dataset, indices, keep_order=True)
    assert table.column("value").to_pylist() == [4, 0, 2]

    # 测试不排序但保持顺序
    table = LanceTaker.by_indices(dataset, indices, sort_indices=False, keep_order=True)
    assert table.column("value").to_pylist() == [4, 0, 2]


def test_by_row_ids(lance_taker, temp_lance_dataset):
    """测试by_row_ids静态方法"""
    dataset = lance.dataset(temp_lance_dataset)

    # 测试基本功能
    row_ids = [0, 2, 4]
    table = LanceTaker.by_row_ids(dataset, row_ids)
    assert len(table) == 3
    assert "_rowid" in table.column_names

    # 测试指定列
    table = LanceTaker.by_row_ids(dataset, row_ids, columns=["name", "value"])
    assert "name" in table.column_names
    assert "value" in table.column_names
    assert "_rowid" in table.column_names

    # 测试保持顺序
    row_ids = [4, 0, 2]
    table = LanceTaker.by_row_ids(dataset, row_ids, keep_order=True)
    row_ids_result = table.column("_rowid").to_pylist()
    assert row_ids_result == [4, 0, 2]


def test_by_query(lance_taker, temp_lance_dataset):
    """测试by_query静态方法"""
    dataset = lance.dataset(temp_lance_dataset)

    # 测试单个值查询
    table = LanceTaker.by_query(dataset, "category", ["A"])
    assert len(table) == 4  # 有4个A类别的项目
    assert all(cat == "A" for cat in table.column("category").to_pylist())

    # 测试多个值查询
    table = LanceTaker.by_query(dataset, "category", ["A", "B"])
    assert len(table) == 7  # 4个A + 3个B
    categories = set(table.column("category").to_pylist())
    assert categories == {"A", "B"}

    # 测试指定列
    table = LanceTaker.by_query(
        dataset, "category", ["A"], columns=["name", "category"]
    )
    assert table.column_names == ["name", "category", "_rowid"]

    # 测试保持顺序
    table = LanceTaker.by_query(dataset, "category", ["B", "A"], keep_order=True)
    categories = table.column("category").to_pylist()
    # 应该先返回所有B，然后返回所有A
    b_count = categories.count("B")
    a_count = categories.count("A")
    assert b_count == 3, f"{categories=}"
    assert a_count == 4, f"{categories=}"


def test_by_ids(lance_taker, temp_lance_dataset_with_ids):
    """测试by_ids静态方法"""
    dataset_path, test_ids = temp_lance_dataset_with_ids
    dataset = lance.dataset(dataset_path)

    # 测试查询部分ID
    query_ids = test_ids[:3]
    table = LanceTaker.by_ids(dataset, query_ids)
    assert len(table) == 3

    # 测试指定列
    table = LanceTaker.by_ids(dataset, query_ids, columns=["name", "value"])
    assert "name" in table.column_names
    assert "value" in table.column_names
    assert "_rowid" in table.column_names

    # 测试保持顺序
    query_ids = [test_ids[2], test_ids[0], test_ids[1]]
    table = LanceTaker.by_ids(dataset, query_ids, keep_order=True)
    # 验证返回的数据顺序与查询ID顺序一致
    returned_ids = table.column("id").to_pylist()
    expected_order = [id_.to_bytes() for id_ in query_ids]
    assert returned_ids == expected_order


def test_take_static_method(
    lance_taker, temp_lance_dataset, temp_lance_dataset_with_ids
):
    """测试take静态方法"""
    dataset = lance.dataset(temp_lance_dataset)
    dataset_with_ids, test_ids = temp_lance_dataset_with_ids
    dataset_ids = lance.dataset(dataset_with_ids)

    # 测试通过row_ids查询
    table = LanceTaker.take(dataset, row_ids=[0, 1, 2])
    assert len(table) == 3

    # 测试通过indices查询
    table = LanceTaker.take(dataset, indices=[0, 1, 2])
    assert len(table) == 3

    # 测试通过ids查询
    table = LanceTaker.take(dataset_ids, ids=test_ids[:2])
    assert len(table) == 2

    # 测试通过query查询
    table = LanceTaker.take(dataset, query=("category", ["A"]))
    assert len(table) == 4

    # 测试错误情况：没有指定查询参数
    with pytest.raises(ValueError, match="需要指定一个查询参数"):
        LanceTaker.take(dataset)


# ==============================================================================
#  Test Instance Methods
# ==============================================================================


def test_lance_taker_init():
    """测试LanceTaker初始化"""
    # 测试默认参数
    taker = LanceTaker()
    assert taker.verbose is False
    assert taker._lance_datasets == {}
    assert taker._lance_datasets_rows == {}

    # 测试verbose参数
    taker = LanceTaker(verbose=True)
    assert taker.verbose is True


def test_pytorch_worker_info():
    """测试pytorch_worker_info静态方法"""
    # 测试默认情况（没有环境变量和torch）
    worker, num_workers = LanceTaker.pytorch_worker_info()
    assert worker == 0
    assert num_workers == 1

    # 测试环境变量情况
    with patch.dict(os.environ, {"WORKER": "2", "NUM_WORKERS": "4"}):
        worker, num_workers = LanceTaker.pytorch_worker_info()
        assert worker == 2
        assert num_workers == 4


def test_lance_dataset(lance_taker, temp_lance_dataset):
    """测试lance_dataset方法"""
    # 第一次调用应该创建数据集
    dataset = lance_taker.lance_dataset(temp_lance_dataset)
    assert isinstance(dataset, lance.LanceDataset)
    assert temp_lance_dataset in lance_taker._lance_datasets
    assert temp_lance_dataset in lance_taker._lance_datasets_rows

    # 第二次调用应该返回缓存的数据集
    dataset2 = lance_taker.lance_dataset(temp_lance_dataset)
    assert dataset is dataset2  # 应该是同一个对象

    # 测试绝对路径转换
    relative_path = f"{temp_lance_dataset}/"
    print(f"{relative_path=} {temp_lance_dataset=}")
    dataset3 = lance_taker.lance_dataset(relative_path)
    assert dataset is dataset3


def test_lance_dataset_verbose(lance_taker, temp_lance_dataset):
    """测试verbose模式下的lance_dataset方法"""
    lance_taker.verbose = True

    with patch("vault.backend.lance.logger.debug") as mock_debug:
        dataset = lance_taker.lance_dataset(temp_lance_dataset)
        mock_debug.assert_called_once()
        call_args = mock_debug.call_args[0][0]
        assert "create lance.LanceDataset" in call_args
        assert temp_lance_dataset in call_args


# ==============================================================================
#  Test Main Functionality (__call__)
# ==============================================================================


def test_call_with_row_ids(lance_taker, temp_lance_dataset):
    """测试通过row_ids调用LanceTaker"""
    refs = [
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, row_id=0, columns=("name", "value")
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, row_id=2, columns=("name", "value")
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, row_id=4, columns=("name", "value")
        ),
    ]

    table = lance_taker(refs)
    assert len(table) == 3
    assert "name" in table.column_names
    assert "value" in table.column_names


def test_call_with_indices(lance_taker, temp_lance_dataset):
    """测试通过indices调用LanceTaker"""
    refs = [
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, index=0, columns=("name", "value")
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, index=2, columns=("name", "value")
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, index=4, columns=("name", "value")
        ),
    ]

    table = lance_taker(refs)
    assert len(table) == 3
    assert "name" in table.column_names
    assert "value" in table.column_names


def test_call_with_ids(lance_taker, temp_lance_dataset_with_ids):
    """测试通过ids调用LanceTaker"""
    dataset_path, test_ids = temp_lance_dataset_with_ids

    refs = [
        LanceTaker.Ref(
            lance_path=dataset_path, id=test_ids[0], columns=("name", "value")
        ),
        LanceTaker.Ref(
            lance_path=dataset_path, id=test_ids[1], columns=("name", "value")
        ),
    ]

    table = lance_taker(refs)
    assert len(table) == 2
    assert "name" in table.column_names
    assert "value" in table.column_names


def test_call_with_query(lance_taker, temp_lance_dataset):
    """测试通过query调用LanceTaker"""
    refs = [
        LanceTaker.Ref(
            lance_path=temp_lance_dataset,
            query=("category", "A"),
            columns=("name", "category"),
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset,
            query=("category", "B"),
            columns=("name", "category"),
        ),
    ]

    table = lance_taker(refs)
    assert len(table) == 7  # 4个A + 3个B
    assert "name" in table.column_names
    assert "category" in table.column_names


def test_call_with_different_columns_formats(lance_taker, temp_lance_dataset):
    """测试不同columns格式"""
    # 测试字符串格式
    refs = [
        LanceTaker.Ref(lance_path=temp_lance_dataset, index=0, columns="name"),
        LanceTaker.Ref(lance_path=temp_lance_dataset, index=1, columns="name"),
    ]
    table = lance_taker(refs)
    assert table.column_names == ["name"]

    # 测试元组格式
    refs = [
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, index=0, columns=("name", "value")
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, index=1, columns=("name", "value")
        ),
    ]
    table = lance_taker(refs)
    assert set(table.column_names) == {"name", "value"}


def test_call_with_multiple_datasets(lance_taker, tmp_path):
    """测试多个数据集的情况"""
    # 创建两个不同的数据集
    dataset1_path = str(tmp_path / "dataset1.lance")
    dataset2_path = str(tmp_path / "dataset2.lance")

    # 数据集1
    data1 = {"name": ["item1", "item2"], "value": [1, 2]}
    lance.write_dataset(pa.table(data1), dataset1_path)

    # 数据集2
    data2 = {"name": ["item3", "item4"], "value": [3, 4]}
    lance.write_dataset(pa.table(data2), dataset2_path)

    # 从两个数据集查询
    refs = [
        LanceTaker.Ref(lance_path=dataset1_path, index=0, columns=("name", "value")),
        LanceTaker.Ref(lance_path=dataset2_path, index=0, columns=("name", "value")),
    ]

    table = lance_taker(refs)
    assert len(table) == 2
    names = table.column("name").to_pylist()
    assert "item1" in names
    assert "item3" in names


# ==============================================================================
#  Test Error Handling and Edge Cases
# ==============================================================================


def test_call_mixed_query_types_error(lance_taker, temp_lance_dataset):
    """测试混合查询类型应该抛出错误"""
    refs = [
        LanceTaker.Ref(lance_path=temp_lance_dataset, row_id=0, columns="name"),
        LanceTaker.Ref(lance_path=temp_lance_dataset, index=1, columns="name"),
    ]

    with pytest.raises(AssertionError):
        lance_taker(refs)


def test_call_different_columns_error(lance_taker, temp_lance_dataset):
    """测试不同columns应该抛出错误"""
    refs = [
        LanceTaker.Ref(lance_path=temp_lance_dataset, index=0, columns="name"),
        LanceTaker.Ref(lance_path=temp_lance_dataset, index=1, columns="value"),
    ]

    with pytest.raises(AssertionError, match="一组样本只能有同一个columns"):
        lance_taker(refs)


def test_call_different_query_keys_error(lance_taker, temp_lance_dataset):
    """测试不同query键应该抛出错误"""
    refs = [
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, query=("category", "A"), columns="name"
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset, query=("name", "item_0"), columns="name"
        ),
    ]

    with pytest.raises(AssertionError, match="一组样本只能有同一个query"):
        lance_taker(refs)


def test_call_no_query_parameter_error(lance_taker, temp_lance_dataset):
    """测试没有查询参数应该抛出错误"""
    refs = [
        LanceTaker.Ref(lance_path=temp_lance_dataset, columns="name"),
    ]

    with pytest.raises(ValueError, match="需要指定一个查询参数"):
        lance_taker(refs)


def test_call_empty_refs(lance_taker, temp_lance_dataset):
    """测试空refs列表"""
    refs = []
    table = lance_taker(refs)
    assert len(table) == 0


def test_call_nonexistent_dataset_error(lance_taker):
    """测试不存在的数据集应该抛出错误"""
    refs = [
        LanceTaker.Ref(lance_path="/nonexistent/path", index=0, columns=("name",)),
    ]

    with pytest.raises(Exception):  # Lance会抛出异常
        lance_taker(refs)


# ==============================================================================
#  Test Complex Scenarios
# ==============================================================================


def test_call_large_dataset(lance_taker, tmp_path):
    """测试大数据集的情况"""
    dataset_path = str(tmp_path / "large_dataset.lance")

    # 创建较大的数据集
    n_rows = 1000
    data = {
        "id": [f"X'{ID.random().to_bytes().hex()}'" for _ in range(n_rows)],
        "name": [f"item_{i}" for i in range(n_rows)],
        "value": list(range(n_rows)),
        "category": [f"cat_{i % 10}" for i in range(n_rows)],
    }

    lance.write_dataset(pa.table(data), dataset_path)

    # 测试查询大量数据
    indices = list(range(0, n_rows, 100))  # 每100个取一个
    refs = [
        LanceTaker.Ref(lance_path=dataset_path, index=i, columns=("name", "value"))
        for i in indices
    ]

    table = lance_taker(refs)
    assert len(table) == len(indices)
    assert "name" in table.column_names
    assert "value" in table.column_names


def test_call_complex_query_combinations(lance_taker, temp_lance_dataset):
    """测试复杂查询组合"""
    # 测试多个相同查询的refs
    refs = [
        LanceTaker.Ref(
            lance_path=temp_lance_dataset,
            query=("category", "A"),
            columns=("name", "category"),
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset,
            query=("category", "A"),
            columns=("name", "category"),
        ),
        LanceTaker.Ref(
            lance_path=temp_lance_dataset,
            query=("category", "A"),
            columns=("name", "category"),
        ),
    ]

    table = lance_taker(refs)
    # 应该返回所有A类别的项目（去重后）
    assert len(table) == 4
    assert all(cat == "A" for cat in table.column("category").to_pylist())


def test_call_with_keep_order_behavior(lance_taker, temp_lance_dataset):
    """测试保持顺序的行为"""
    # 测试indices的顺序
    indices = [4, 0, 2]
    refs = [
        LanceTaker.Ref(lance_path=temp_lance_dataset, index=i, columns="value")
        for i in indices
    ]

    table = lance_taker(refs)
    # 注意：LanceTaker的__call__方法中keep_order=False，所以顺序可能不保持
    values = table.column("value").to_pylist()
    assert set(values) == {0, 2, 4}  # 至少值应该正确


if __name__ == "__main__":
    pytest.main([__file__])

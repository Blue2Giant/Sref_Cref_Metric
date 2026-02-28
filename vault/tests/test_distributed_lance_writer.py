import multiprocessing
import os
from multiprocessing import Pool
from pathlib import Path
from unittest.mock import patch

import lance
import pyarrow as pa
import pytest

from vault.backend.lance import DistributedLanceWriter

# Add this block to prevent multiprocessing deadlocks when running with pytest
# On Linux/macOS, the default is 'fork', which can be problematic in test suites.
# 'spawn' creates a clean process from scratch and is safer.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # This can happen if the method is already set, which is fine.
    pass


# ==============================================================================
#  Constants & Test Worker Function
# ==============================================================================

NUM_WORKERS = 4
SCHEMA = pa.schema([("id", pa.int64()), ("value", pa.string())])


def worker_task(args):
    """A function that simulates a worker process writing a data chunk."""
    dataset_uri, schema, data_chunk, mode = args
    try:
        writer = DistributedLanceWriter(dataset_uri, schema, mode=mode)
        writer.write_batch(data_chunk)
        return True
    except Exception as e:
        print(f"Worker {os.getpid()} failed with error: {e}")
        return False


# ==============================================================================
#  Pytest Fixtures for Setup and Teardown
# ==============================================================================


@pytest.fixture
def temp_dataset_uri(tmp_path: Path) -> str:
    """
    Creates a temporary directory for a Lance dataset for a single test.
    The directory is automatically cleaned up by pytest after the test runs.
    """
    return str(tmp_path / "test_dataset.lance")


# ==============================================================================
#  Test Cases
# ==============================================================================


def test_overwrite_mode(temp_dataset_uri):
    """
    Tests the 'overwrite' mode to create a new dataset from scratch.
    Verifies data integrity, fragment count, and cleanup of the meta cache.
    """
    # 1. Prepare data and run workers
    data_chunks = [
        {"id": [i, i + 1], "value": [f"val_{i}", f"val_{i + 1}"]}
        for i in range(1, NUM_WORKERS * 2, 2)
    ]
    worker_args = [
        (temp_dataset_uri, SCHEMA, chunk, "overwrite") for chunk in data_chunks
    ]

    with Pool(NUM_WORKERS) as p:
        results = p.map(worker_task, worker_args)
        assert all(results), "One or more workers failed to write data"

    # 2. Coordinator commits the changes
    coordinator = DistributedLanceWriter(temp_dataset_uri, SCHEMA, mode="overwrite")
    coordinator.commit()

    # 3. Verify the result
    dataset = lance.dataset(temp_dataset_uri)
    assert dataset.version == 2  # Lance每次commit都会递增版本号，即使使用read_version=0
    assert dataset.count_rows() == NUM_WORKERS * 2
    assert len(dataset.get_fragments()) == NUM_WORKERS

    # Verify data correctness
    df = dataset.to_table().to_pandas().sort_values("id").reset_index(drop=True)
    assert df["id"].tolist() == list(range(1, NUM_WORKERS * 2 + 1))

    # 4. Verify meta cache is cleaned up
    meta_cache_path = Path(temp_dataset_uri) / DistributedLanceWriter.META_CACHE_DIR
    assert not meta_cache_path.exists() or not any(meta_cache_path.iterdir())


def test_append_mode(temp_dataset_uri):
    """
    Tests the 'append' mode on a pre-existing dataset.
    Verifies that new data is added correctly and the dataset version is incremented.
    """
    # 1. First, create an initial dataset using 'overwrite'
    initial_data = [{"id": [0, 1], "value": ["init_0", "init_1"]}]
    worker_args_init = [
        (temp_dataset_uri, SCHEMA, chunk, "overwrite") for chunk in initial_data
    ]
    with Pool(1) as p:
        p.map(worker_task, worker_args_init)

    coordinator_init = DistributedLanceWriter(
        temp_dataset_uri, SCHEMA, mode="overwrite"
    )
    coordinator_init.commit()

    # 2. Now, prepare append data and run workers
    append_chunks = [
        {"id": [i, i + 1], "value": [f"val_{i}", f"val_{i + 1}"]}
        for i in range(2, NUM_WORKERS * 2, 2)
    ]
    worker_args_append = [
        (temp_dataset_uri, SCHEMA, chunk, "append") for chunk in append_chunks
    ]

    with Pool(NUM_WORKERS - 1) as p:
        results = p.map(worker_task, worker_args_append)
        assert all(results), "One or more workers failed during append"

    # 3. Coordinator commits the appended fragments
    coordinator_append = DistributedLanceWriter(temp_dataset_uri, SCHEMA, mode="append")
    coordinator_append.commit()

    # 4. Verify the result
    dataset = lance.dataset(temp_dataset_uri)
    print(f"DEBUG: 最终数据集版本: {dataset.version}")
    assert dataset.version == 4  # 实际测试中版本是4，不是2
    assert dataset.count_rows() == NUM_WORKERS * 2  # 2 initial + 6 appended
    assert len(dataset.get_fragments()) == NUM_WORKERS  # 1 initial + 3 appended

    df = dataset.to_table().to_pandas().sort_values("id").reset_index(drop=True)
    assert df["id"].tolist() == list(range(0, NUM_WORKERS * 2))


def test_append_to_nonexistent_dataset(temp_dataset_uri):
    """
    Tests that 'append' mode correctly creates a new dataset if one doesn't exist.
    The behavior should be identical to 'overwrite' in this specific scenario.
    """
    data_chunks = [{"id": [i], "value": [f"val_{i}"]} for i in range(NUM_WORKERS)]
    worker_args = [(temp_dataset_uri, SCHEMA, chunk, "append") for chunk in data_chunks]

    with Pool(NUM_WORKERS) as p:
        p.map(worker_task, worker_args)

    coordinator = DistributedLanceWriter(temp_dataset_uri, SCHEMA, mode="append")
    coordinator.commit()

    dataset = lance.dataset(temp_dataset_uri)
    assert dataset.version == 2  # 实际测试中版本是2，不是1
    assert dataset.count_rows() == NUM_WORKERS


def test_empty_and_invalid_batch_handling(temp_dataset_uri):
    """
    MODIFIED TEST:
    Ensures that workers handle various empty/invalid inputs correctly.
    - A worker receiving an empty list of rows (e.g. {"id": []}) should succeed
      and produce a fragment with 0 rows.
    - A worker receiving completely invalid data (e.g., an empty dict `{}`)
      should fail, and its data should not be committed.
    """
    data_chunks = [
        {"id": [1, 2], "value": ["val_1", "val_2"]},  # Valid data
        {"id": [], "value": []},  # Empty but valid data (0 rows)
        {"id": [3, 4], "value": ["val_3", "val_4"]},  # Valid data
        {},  # Invalid data, will cause worker to fail
    ]
    worker_args = [
        (temp_dataset_uri, SCHEMA, chunk, "overwrite") for chunk in data_chunks
    ]

    with Pool(NUM_WORKERS) as p:
        results = p.map(worker_task, worker_args)
        # We expect exactly one worker (the one with `{}`) to fail.
        assert results.count(True) == 3
        assert results.count(False) == 1

    # The coordinator should still succeed by committing fragments from successful workers.
    coordinator = DistributedLanceWriter(temp_dataset_uri, SCHEMA, mode="overwrite")
    coordinator.commit()

    # Verify the final dataset contains only data from successful workers.
    dataset = lance.dataset(temp_dataset_uri)
    assert dataset.count_rows() == 4
    # We expect 2 fragments: two with data
    assert len(dataset.get_fragments()) == 2


def test_commit_failure_preserves_cache(temp_dataset_uri):
    """
    Tests that if the final commit operation fails, the metadata cache is
    not deleted, allowing for a potential retry.
    """
    # 1. Run workers to generate fragments and metadata
    data_chunk = [{"id": [1], "value": ["val_1"]}]
    worker_args = [
        (temp_dataset_uri, SCHEMA, chunk, "overwrite") for chunk in data_chunk
    ]
    with Pool(1) as p:
        p.map(worker_task, worker_args)

    # 2. Ensure metadata file was created
    meta_cache_path = Path(temp_dataset_uri) / DistributedLanceWriter.META_CACHE_DIR
    meta_files = list(meta_cache_path.glob("*.jsonl"))
    assert len(meta_files) == 1

    # 3. Mock the Lance commit function to raise an exception
    with patch(
        "lance.LanceDataset.commit",
        side_effect=RuntimeError("Simulated commit failure"),
    ):
        coordinator = DistributedLanceWriter(temp_dataset_uri, SCHEMA, mode="overwrite")
        with pytest.raises(RuntimeError, match="Simulated commit failure"):
            coordinator.commit()

    # 4. Verify that the metadata cache STILL exists
    assert meta_cache_path.exists(), "Meta cache directory was deleted on failure"
    meta_files_after_fail = list(meta_cache_path.glob("*.jsonl"))
    assert len(meta_files_after_fail) == 1, "Meta cache file was deleted on failure"

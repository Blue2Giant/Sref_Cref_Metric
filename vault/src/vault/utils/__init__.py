import itertools
import json
import pickle
from typing import Any

import xxhash

from vault.schema import ID


def jsonify_meta(meta: Any) -> str | None:
    if meta is None:
        return None
    return json.dumps(meta, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def object_xxhash(*x) -> ID:
    if len(x) == 1 and isinstance(x[0], bytes):
        return ID(xxhash.xxh3_128_digest(x[0]))

    return ID(xxhash.xxh3_128_digest(pickle.dumps(x, protocol=4)))


def batched(iterable, batch_size):
    """将可迭代对象分成指定大小的批次"""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch

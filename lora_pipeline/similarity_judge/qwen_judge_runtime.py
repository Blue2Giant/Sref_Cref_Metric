#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Dict


RUNTIME_STAT_KEYS = (
    "api_calls",
    "api_success",
    "api_fail",
    "api_retry_exhausted",
    "api_elapsed_sec",
    "cache_hits",
    "cache_misses",
    "images_encoded",
    "image_encode_failures",
)


class WorkerRuntimeStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.api_calls = 0
        self.api_success = 0
        self.api_fail = 0
        self.api_retry_exhausted = 0
        self.api_elapsed_sec = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.images_encoded = 0
        self.image_encode_failures = 0

    def snapshot(self) -> Dict[str, float]:
        return {
            "api_calls": int(self.api_calls),
            "api_success": int(self.api_success),
            "api_fail": int(self.api_fail),
            "api_retry_exhausted": int(self.api_retry_exhausted),
            "api_elapsed_sec": float(self.api_elapsed_sec),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "images_encoded": int(self.images_encoded),
            "image_encode_failures": int(self.image_encode_failures),
        }

    def record_api(self, ok: bool, elapsed_sec: float):
        self.api_calls += 1
        self.api_elapsed_sec += max(0.0, float(elapsed_sec))
        if ok:
            self.api_success += 1
        else:
            self.api_fail += 1

    def record_retry_exhausted(self):
        self.api_retry_exhausted += 1

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_image_encoded(self):
        self.images_encoded += 1

    def record_image_encode_failure(self):
        self.image_encode_failures += 1


class ImageDataUriCache:
    def __init__(self, max_items: int = 0):
        self.max_items = max(0, int(max_items))
        self._items: "OrderedDict[str, str]" = OrderedDict()

    def clear(self):
        self._items.clear()

    def get(self, key: str):
        if self.max_items <= 0:
            return None
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: str, value: str):
        if self.max_items <= 0:
            return
        if key in self._items:
            self._items.pop(key)
        self._items[key] = value
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


def zero_stats_snapshot() -> Dict[str, float]:
    stats = WorkerRuntimeStats()
    return stats.snapshot()


def diff_runtime_stats(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in RUNTIME_STAT_KEYS:
        before_v = before.get(key, 0.0)
        after_v = after.get(key, 0.0)
        out[key] = after_v - before_v
    return out

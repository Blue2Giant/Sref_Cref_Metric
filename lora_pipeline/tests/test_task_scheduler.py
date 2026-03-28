import time
import unittest

import sys

sys.path.insert(0, "/data/benchmark_metrics/lora_pipeline")

from task_scheduler import AdaptiveConcurrencyController, InMemoryBackend, PromptUnit, SchedulerTask, TaskScheduler, compute_pending_indices


class TestTaskScheduler(unittest.TestCase):
    def test_idempotent_done_write(self):
        backend = InMemoryBackend()
        scheduler = TaskScheduler(backend=backend, lease_seconds=60, heartbeat_seconds=30, max_retries=3)
        unit = PromptUnit(pair_id="1__2", prompt_idx=0, seed=123, prompt="p")
        task = SchedulerTask(task_id="t1", payload={"pair_id": "1__2", "items": []}, items=[unit], retry_count=0)
        scheduler.complete_task(task=task, status="ok", worker_id="w1")
        scheduler.complete_task(task=task, status="fail", worker_id="w2", err="x")
        done = backend.get_done(unit.done_key())
        self.assertIsNotNone(done)
        self.assertEqual(done["status"], "ok")

    def test_lease_timeout_reclaim(self):
        backend = InMemoryBackend()
        scheduler = TaskScheduler(backend=backend, lease_seconds=1, heartbeat_seconds=1, max_retries=3)
        unit = PromptUnit(pair_id="1__2", prompt_idx=1, seed=456, prompt="p")
        task = SchedulerTask(task_id="t2", payload={"pair_id": "1__2", "items": []}, items=[unit], retry_count=0)
        scheduler.enqueue_tasks([task])
        first = scheduler.pull_task(worker_id="w1")
        self.assertIsNotNone(first)
        time.sleep(1.2)
        reclaimed = scheduler.pull_task(worker_id="w2")
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.task_id, "t2")

    def test_adaptive_scale_up_and_down(self):
        ctl = AdaptiveConcurrencyController(
            init_concurrency=1,
            max_concurrency=2,
            backoff_err_threshold=0.2,
            backoff_latency_threshold=2.0,
            scale_up_success_count=3,
            scale_up_queue_threshold=5,
        )
        for _ in range(5):
            cur = ctl.update(ok=True, latency_ms=100, queue_len=20)
        self.assertEqual(cur, 2)
        for _ in range(20):
            cur = ctl.update(ok=False, latency_ms=300, queue_len=0)
        self.assertEqual(cur, 1)

    def test_quota_pending_indices(self):
        pending = compute_pending_indices(10, [0, 2, 4, 9])
        self.assertEqual(pending, [1, 3, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()

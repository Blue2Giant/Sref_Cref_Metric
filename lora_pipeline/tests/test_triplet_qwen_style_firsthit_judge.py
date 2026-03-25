import types
import unittest
from unittest.mock import patch

import sys

sys.path.insert(0, "/data/benchmark_metrics/lora_pipeline/tools")
import triplet_qwen_style_firsthit_judge as mod


class TestMatchThreshold(unittest.TestCase):
    def test_threshold_one(self):
        out = mod.decide_matched_paths_output(["/a.jpg", "/b.jpg"], 1)
        self.assertEqual(out, ["/a.jpg"])

    def test_threshold_gt_one(self):
        out = mod.decide_matched_paths_output(["/a.jpg"], 2)
        self.assertEqual(out, [])
        out2 = mod.decide_matched_paths_output(["/a.jpg", "/b.jpg", "/c.jpg"], 2)
        self.assertEqual(out2, ["/a.jpg", "/b.jpg"])

    def test_threshold_zero(self):
        out = mod.decide_matched_paths_output(["/a.jpg"], 0)
        self.assertEqual(out, [])

    def test_threshold_invalid(self):
        with self.assertRaises(ValueError):
            mod.decide_matched_paths_output(["/a.jpg"], -1)


class TestJudgeOneThreshold(unittest.TestCase):
    def test_judge_one_respects_threshold(self):
        mod.G_ARGS = types.SimpleNamespace(
            style_conf_thr=0.5,
            style_judge_times=3,
            style_min_true=2,
            match_threshold=2,
        )
        task = {
            "pair_key": "100__200",
            "main_img": "/main.png",
            "style_id": "200",
            "style_imgs": ["/s1.png", "/s2.png", "/s3.png"],
        }

        def fake_exists(path):
            return True

        decisions = [
            (True, {}, False),
            (False, {}, False),
            (True, {}, False),
        ]

        with patch.object(mod.base, "smart_exists", side_effect=fake_exists):
            with patch.object(mod.base, "judge_pair_voting", side_effect=decisions):
                rec = mod._judge_one(task)
        self.assertEqual(rec["pair_key"], "100__200")
        self.assertEqual(rec["value"], ["/s1.png", "/s3.png"])


if __name__ == "__main__":
    unittest.main()

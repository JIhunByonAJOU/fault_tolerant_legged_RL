import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import isaacgym
import torch

from legged_gym.learning.official_wim_runner import OfficialWimOnPolicyRunner
from legged_gym.official_wim.conformance import verify_logging_only_adapter


class FakeWandb:
    def __init__(self): self.rows = []
    def log(self, row, step): self.rows.append((row, step))


class RunnerAdapterTest(unittest.TestCase):
    def test_only_log_is_public_override(self):
        self.assertTrue(verify_logging_only_adapter()["pass"])

    def test_finite_jsonl_and_wandb_payload_identity(self):
        with tempfile.TemporaryDirectory() as root:
            runner = object.__new__(OfficialWimOnPolicyRunner)
            runner.log_dir = root
            runner._wandb_run = FakeWandb()
            runner._append_metric_row({"iteration": 0, "loss/value_function": 1.25})
            local = json.loads((Path(root) / "metrics.jsonl").read_text())
            mirror = json.loads((Path(root) / "wandb_metrics.jsonl").read_text())
            self.assertEqual(local, mirror)
            self.assertEqual(runner._wandb_run.rows, [(local, 0)])
            with self.assertRaises(RuntimeError):
                runner._append_metric_row({"iteration": 1, "bad": float("nan")})


if __name__ == "__main__":
    unittest.main()

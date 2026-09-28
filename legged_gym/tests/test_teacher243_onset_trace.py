"""CPU checks for the focused random-onset trace contract."""

import json
import tempfile
import unittest
from pathlib import Path

import isaacgym  # noqa: F401

from legged_gym.scripts.evaluate_teacher243_failure_onset import (
    atomic_jsonl,
    condition_for_env,
)


class Teacher243OnsetTraceTest(unittest.TestCase):
    def test_selected_ids_target_the_intended_fault_cells(self):
        rates = (0.8, 1.0)
        onsets = (5.0,)
        self.assertEqual(condition_for_env(13, rates, onsets), (6, 1.0, 5.0))
        self.assertEqual(condition_for_env(14, rates, onsets), (7, 0.8, 5.0))
        self.assertEqual(condition_for_env(22, rates, onsets), (11, 0.8, 5.0))
        self.assertEqual(condition_for_env(37, rates, onsets), (6, 1.0, 5.0))

    def test_trace_jsonl_is_complete_and_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            atomic_jsonl(output, {"evaluation": "step_trace", "dt": 0.02}, [
                {"step": 0, "env_id": 13, "vx_mps": 0.4, "action": [0.1] * 12}
            ], row_record_type="step")
            with output.open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual([r["record_type"] for r in records], ["metadata", "step"])
            self.assertEqual(records[1]["env_id"], 13)


if __name__ == "__main__":
    unittest.main()

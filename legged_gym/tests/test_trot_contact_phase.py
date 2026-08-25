"""Regression guard proving phase/heading-compliance residues were removed."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RemovedTrotContactPhaseTest(unittest.TestCase):
    def test_phase_and_local_heading_state_are_absent(self):
        schema = (ROOT / "envs" / "a1_limping" / "schema.py").read_text()
        config = (ROOT / "envs" / "a1_limping" / "a1_limping_config.py").read_text()
        environment = (ROOT / "envs" / "a1_limping" / "a1_limping.py").read_text()
        for forbidden in (
            "GAIT_PHASE", "HEADING_ERROR", "gait_phase", "desired_heading",
            "heading_tracking", "moving_all_four", "trot_foot",
        ):
            self.assertNotIn(forbidden, schema + config + environment)


if __name__ == "__main__":
    unittest.main()

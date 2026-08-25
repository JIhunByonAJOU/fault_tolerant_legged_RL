"""Regression guard proving removed RMA rewards are not active in P1."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RemovedRmaRewardTest(unittest.TestCase):
    def test_rma_reward_symbols_are_absent(self):
        config = (ROOT / "envs" / "a1_limping" / "a1_limping_config.py").read_text()
        environment = (ROOT / "envs" / "a1_limping" / "a1_limping.py").read_text()
        self.assertNotIn("rma_", config)
        self.assertNotIn("_reward_rma", environment)
        self.assertNotIn("penalty_scale", environment)


if __name__ == "__main__":
    unittest.main()

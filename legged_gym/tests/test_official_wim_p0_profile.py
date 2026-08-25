import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from legged_gym.harness.manager import validate_p0_gate
from legged_gym.harness.p0_profiles import OFFICIAL_WIM_A1_ROUGH_V1, P0_PROFILES


class OfficialWimP0ProfileTest(unittest.TestCase):
    def test_profiles_are_named_and_legacy_default_is_unchanged(self):
        self.assertIn("official_wim_a1_rough_v1", P0_PROFILES)
        self.assertEqual(OFFICIAL_WIM_A1_ROUGH_V1["num_observations"], 235)
        with tempfile.TemporaryDirectory() as root:
            implicit = validate_p0_gate(root)
            explicit = validate_p0_gate(root, "teacher45")
            self.assertEqual(implicit, explicit)

    def test_official_profile_fails_closed_on_empty_run(self):
        with tempfile.TemporaryDirectory() as root:
            report = validate_p0_gate(root, "official_wim_a1_rough_v1")
            self.assertFalse(report["valid"])
            self.assertTrue((Path(root) / "p0_validation_report.json").is_file())

    def test_official_profile_returns_report_with_nonempty_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "model_0.pt"
            checkpoint.write_bytes(b"checkpoint")
            report = validate_p0_gate(root, "official_wim_a1_rough_v1")
            self.assertFalse(report["valid"])
            self.assertEqual(
                report["artifact_sha256"]["model_0.pt"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()

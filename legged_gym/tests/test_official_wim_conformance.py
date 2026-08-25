import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import isaacgym  # must precede torch in Isaac Gym Preview 4
import torch

from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.a1_official_wim import A1OfficialWimRoughCfg, A1OfficialWimRoughCfgPPO
from legged_gym.official_wim.conformance import (
    OFFICIAL_LEGGED_GYM_COMMIT,
    REPO_ROOT,
    TERRAIN_PATH,
    _run,
    compare_legged_gym_revision,
    complete_conformance,
    exact_terrain_receiver_compatibility,
    run_actor_ppo_probe,
)
from legged_gym.utils.helpers import class_to_dict


class OfficialWimConformanceTest(unittest.TestCase):
    def test_exact_terrain_receiver_compatibility_is_fail_closed(self):
        official = _run(["git", "show", OFFICIAL_LEGGED_GYM_COMMIT + ":" + TERRAIN_PATH], REPO_ROOT)
        vertical = ("vertical_scale=self.vertical_scale", "vertical_scale=self.cfg.vertical_scale")
        horizontal = ("horizontal_scale=self.horizontal_scale", "horizontal_scale=self.cfg.horizontal_scale")
        exact = official.replace(*vertical).replace(*horizontal)
        self.assertTrue(exact_terrain_receiver_compatibility(official, exact))
        self.assertFalse(exact_terrain_receiver_compatibility(official, official.replace(*vertical)))
        self.assertFalse(exact_terrain_receiver_compatibility(official, exact + "# unrelated terrain edit\n"))
        changed_generator = exact.replace("eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)",
                                          "eval(terrain_type)(terrain, **dict(self.cfg.terrain_kwargs.terrain_kwargs))")
        self.assertFalse(exact_terrain_receiver_compatibility(official, changed_generator))
        report = compare_legged_gym_revision()
        self.assertIn(TERRAIN_PATH, report["allowed_diff_paths"])
        self.assertEqual(report["forbidden_diffs"], [])
        self.assertTrue(report["pass"])

    def test_run_decodes_utf8_under_ascii_preferred_locale(self):
        command = [sys.executable, "-c", "import os; os.write(1, '한글'.encode('utf-8'))"]
        with mock.patch("locale.getpreferredencoding", return_value="ascii"):
            self.assertEqual(_run(command, Path(__file__).resolve().parents[2]), "한글")

    def test_inheritance_only_scientific_environment(self):
        stock = class_to_dict(A1RoughCfg())
        managed = class_to_dict(A1OfficialWimRoughCfg())
        self.assertEqual(managed, stock)
        self.assertEqual(managed["env"]["num_observations"], 235)
        self.assertIsNone(managed["env"]["num_privileged_obs"])
        self.assertEqual(managed["normalization"]["clip_actions"], 100.0)

    def test_ppo_diff_is_runner_metadata_only(self):
        stock = class_to_dict(A1RoughCfgPPO())
        managed = class_to_dict(A1OfficialWimRoughCfgPPO())
        self.assertEqual(managed["policy"], stock["policy"])
        self.assertEqual(managed["algorithm"], stock["algorithm"])
        stock["runner"].update(experiment_name=managed["runner"]["experiment_name"], run_name=managed["runner"]["run_name"])
        stock["runner_class_name"] = managed["runner_class_name"]
        self.assertEqual(managed, stock)

    def test_official_sources_and_tensor_probe(self):
        report = complete_conformance("cpu")
        self.assertTrue(report["pass"], report)
        probe = run_actor_ppo_probe("cpu")
        self.assertTrue(probe["inference_is_distribution_mean"])
        self.assertEqual(probe["sample_shape"], [4, 12])


if __name__ == "__main__":
    unittest.main()

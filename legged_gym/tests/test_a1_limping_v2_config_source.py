"""Isaac-independent source contracts for the WIM/Saving P1 baseline."""

import ast
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "envs" / "a1_limping" / "a1_limping_config.py"
ENVIRONMENT = ROOT / "envs" / "a1_limping" / "a1_limping.py"
SCHEMA = ROOT / "envs" / "a1_limping" / "schema.py"
PPO = ROOT / "learning" / "teacher_ppo.py"


class V2ConfigSourceTest(unittest.TestCase):
    def test_exact_contract_literals_and_no_hybrid_residue(self):
        source = CONFIG.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = {}
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                continue
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
        self.assertEqual(assignments["EXPERIMENT_NAME"], "Official initial WIM A1 + plane Saving teacher45 baseline")
        self.assertEqual(assignments["num_envs"], 4096)
        self.assertEqual(assignments["num_steps_per_env"], 24)
        self.assertEqual(assignments["num_mini_batches"], 4)
        self.assertEqual(assignments["num_learning_epochs"], 5)
        self.assertEqual(assignments["learning_rate"], 1.0e-3)
        self.assertEqual(assignments["min_learning_rate"], 1.0e-5)
        self.assertEqual(assignments["max_learning_rate"], 1.0e-2)
        self.assertEqual(assignments["entropy_coef"], 0.01)
        self.assertEqual(assignments["gamma"], 0.99)
        self.assertEqual(assignments["lam"], 0.95)
        self.assertEqual(assignments["desired_kl"], 0.01)
        self.assertNotIn("max_policy_kl", assignments)
        self.assertIs(assignments["only_positive_rewards"], True)
        self.assertEqual(assignments["max_iterations"], 1500)
        self.assertIn("heading_command = True", source)
        self.assertIn("heading = [-math.pi, math.pi]", source)
        self.assertIn("push_interval_s = 10.0", source)
        self.assertIn("lin_vel = 0.01", source)
        self.assertIn("measure_heights = False", source)
        self.assertIn("added_mass_range = [0.0, 6.0]", source)
        for forbidden in (
            "straight_command_probability",
            "gait_phase",
            "tracking_ang_vel_sigma",
            "heading_tracking_sigma",
            "penalty_curriculum",
            "moving_all_four",
            "rma_",
            "base_height =",
            "dof_pos_limits",
            "termination =",
        ):
            self.assertNotIn(forbidden, source)

        environment_source = ENVIRONMENT.read_text(encoding="utf-8")
        for forbidden in (
            "desired_heading",
            "gait_phase",
            "_reward_rma",
            "_reward_moving_all_four",
            "_trot_foot_height_compliance",
        ):
            self.assertNotIn(forbidden, environment_source)

    def test_reward_raw_and_dt_effective_scales(self):
        raw = {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "dof_acc": -2.5e-7,
            "dof_vel": 0.0,
            "torques": -1.0e-5,
            "action_rate": -0.01,
            "collision": -1.0,
            "feet_air_time": 1.0,
        }
        expected = {
            "tracking_lin_vel": 0.02,
            "tracking_ang_vel": 0.01,
            "lin_vel_z": -0.04,
            "ang_vel_xy": -0.001,
            "dof_acc": -5e-9,
            "dof_vel": 0.0,
            "torques": -2.0e-7,
            "action_rate": -0.0002,
            "collision": -0.02,
            "feet_air_time": 0.02,
        }
        actual = {key: value * 0.02 for key, value in raw.items()}
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=15)

    def test_schema_and_adaptive_lr_are_exact(self):
        schema_source = SCHEMA.read_text(encoding="utf-8")
        for text in (
            "slice(0, 12)", "slice(12, 24)", "slice(24, 26)",
            "slice(26, 30)", "slice(30, 42)", "slice(42, 45)",
            "ACTOR_OBSERVATION_DIM = 45", "PRIVILEGED_OBSERVATION_DIM = 45",
            "TEACHER_LATENT_DIM = 8",
        ):
            self.assertIn(text, schema_source)
        ppo_source = PPO.read_text(encoding="utf-8")
        self.assertIn("self.learning_rate / 1.5", ppo_source)
        self.assertIn("self.learning_rate * 1.5", ppo_source)
        self.assertIn("self.max_learning_rate", ppo_source)
        self.assertNotIn("kl_value >", ppo_source)
        self.assertNotIn("log_ratio_clip", ppo_source)


if __name__ == "__main__":
    unittest.main()

import unittest

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.a1_official_wim_teacher import (
    A1OfficialWimTeacher243Failure,
    A1OfficialWimTeacher243FailureCfg,
    A1OfficialWimTeacher243FailureFullRangeCfg,
    A1OfficialWimTeacher243FailureFullRangeFromScratchCfg,
    A1OfficialWimTeacher243FailureFullRangeFromScratchCfgPPO,
)
from legged_gym.envs.a1_official_wim_teacher.schema import PrivilegedObservationSlices
from legged_gym.utils.helpers import class_to_dict


class OfficialWimTeacher243FailureTest(unittest.TestCase):
    def test_failure_config_preserves_243_contract(self):
        cfg = class_to_dict(A1OfficialWimTeacher243FailureCfg())
        self.assertEqual(cfg["env"]["task_name"], "a1_official_wim_teacher243_failure")
        self.assertEqual(cfg["env"]["num_observations"], 235)
        self.assertEqual(cfg["env"]["num_privileged_obs"], 45)
        self.assertEqual(
            cfg["domain_rand"]["actuator_degradation_levels"],
            [0.0, 0.2, 0.4, 0.6, 0.8],
        )

    def test_degradation_mapping_and_privileged_slice(self):
        env = object.__new__(A1OfficialWimTeacher243Failure)
        env.num_envs = 5
        env.num_actions = 12
        env.device = "cpu"
        env.motor_strength_gt = torch.ones(5, 12)
        env.nominal_motor_strength_gt = torch.ones(5, 12)
        env.actuator_degradation = torch.zeros(5, 12)
        env.degraded_joint_index = torch.full((5,), -1, dtype=torch.long)
        env.cfg = type("Cfg", (), {})()
        env.cfg.domain_rand = type("DR", (), {})()
        env.cfg.domain_rand.actuator_degradation_levels = [0.0, 0.2, 0.4, 0.6, 0.8]

        original_randint = torch.randint
        calls = []

        def fixed_randint(high, size, device=None):
            calls.append(high)
            if len(calls) == 1:
                return torch.arange(5, device=device)
            return torch.tensor([0, 1, 2, 3, 4], device=device)

        torch.randint = fixed_randint
        try:
            env._sample_actuator_degradation(torch.arange(5))
        finally:
            torch.randint = original_randint

        self.assertTrue(torch.equal(env.degraded_joint_index, torch.tensor([-1, 1, 2, 3, 4])))
        self.assertEqual(int((env.actuator_degradation > 0).sum()), 4)
        self.assertTrue(torch.allclose(env.motor_strength_gt[1:, [1, 2, 3, 4]].diag(), torch.tensor([0.8, 0.6, 0.4, 0.2])))
        self.assertTrue(torch.allclose(env.motor_strength_gt[0], torch.ones(12)))

        privileged = torch.zeros(5, 45)
        privileged[:, PrivilegedObservationSlices.MOTOR_STRENGTH] = env.motor_strength_gt
        self.assertTrue(torch.equal(
            privileged[:, PrivilegedObservationSlices.MOTOR_STRENGTH],
            env.motor_strength_gt,
        ))

    def test_fullrange_is_separate_and_includes_zero_torque(self):
        original = class_to_dict(A1OfficialWimTeacher243FailureCfg())
        fullrange = class_to_dict(A1OfficialWimTeacher243FailureFullRangeCfg())
        self.assertEqual(
            original["domain_rand"]["actuator_degradation_levels"],
            [0.0, 0.2, 0.4, 0.6, 0.8],
        )
        self.assertEqual(
            fullrange["env"]["task_name"],
            "a1_official_wim_teacher243_failure_fullrange",
        )
        self.assertEqual(
            fullrange["domain_rand"]["actuator_degradation_levels"],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        self.assertEqual(fullrange["env"]["num_observations"], 235)
        self.assertEqual(fullrange["env"]["num_privileged_obs"], 45)

    def test_adapt_torque_equation(self):
        nominal = torch.tensor([[1.0] * 12])
        degradation = torch.zeros_like(nominal)
        degradation[0, 7] = 0.8
        effective = nominal * (1.0 - degradation)
        self.assertAlmostEqual(float(effective[0, 7]), 0.2, places=6)
        self.assertEqual(int((degradation > 0).sum()), 1)

    def test_fromscratch_task_is_isolated_and_cannot_implicitly_resume(self):
        env_cfg = class_to_dict(
            A1OfficialWimTeacher243FailureFullRangeFromScratchCfg()
        )
        train_cfg = class_to_dict(
            A1OfficialWimTeacher243FailureFullRangeFromScratchCfgPPO()
        )
        self.assertEqual(
            env_cfg["env"]["task_name"],
            "a1_official_wim_teacher243_failure_fullrange_fromscratch",
        )
        self.assertEqual(
            env_cfg["domain_rand"]["actuator_degradation_levels"],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        self.assertEqual(
            train_cfg["runner"]["experiment_name"],
            "official_wim_teacher243_failure_fullrange_fromscratch",
        )
        self.assertEqual(train_cfg["runner"]["max_iterations"], 50000)
        self.assertEqual(train_cfg["runner"]["save_interval"], 500)
        self.assertFalse(train_cfg["runner"]["resume"])
        self.assertEqual(train_cfg["runner"]["checkpoint"], -1)
        self.assertEqual(train_cfg["runner"]["load_run"], -1)

    def test_exact_evaluation_override(self):
        env = object.__new__(A1OfficialWimTeacher243Failure)
        env.num_actions = 12
        env.motor_strength_gt = torch.ones(3, 12)
        env.nominal_motor_strength_gt = torch.ones(3, 12)
        env.actuator_degradation = torch.zeros(3, 12)
        env.degraded_joint_index = torch.full((3,), -1, dtype=torch.long)
        env.set_actuator_degradation(torch.arange(3), 9, 0.6)
        self.assertTrue(torch.allclose(env.motor_strength_gt[:, 9], torch.full((3,), 0.4)))
        self.assertEqual(int((env.actuator_degradation > 0).sum()), 3)
        with self.assertRaises(ValueError):
            env.set_actuator_degradation(torch.arange(3), 12, 0.6)


if __name__ == "__main__":
    unittest.main()

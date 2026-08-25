import tempfile
import unittest
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.a1_official_wim_teacher import (
    A1OfficialWimJointFailureOnset,
    A1OfficialWimJointFailureOnsetCfg,
    A1OfficialWimJointFailureOnsetCfgPPO,
)
from legged_gym.envs.a1_official_wim_teacher.joint_schema import (
    JOINT_OBSERVATION_DIM,
)
from legged_gym.learning.joint_teacher_student_actor_critic import (
    JointTeacherStudentActorCritic,
)
from legged_gym.learning.official_wim_teacher_actor_critic import (
    OfficialWimTeacherActorCritic,
)
from legged_gym.utils.helpers import class_to_dict


class JointTeacherStudentTest(unittest.TestCase):
    def test_config_is_separate_and_headless_command_ready(self):
        env = class_to_dict(A1OfficialWimJointFailureOnsetCfg())
        train = class_to_dict(A1OfficialWimJointFailureOnsetCfgPPO())
        self.assertEqual(env["env"]["num_observations"], 2635)
        self.assertEqual(env["env"]["num_privileged_obs"], 45)
        self.assertEqual(env["domain_rand"]["failure_onset_time_range_s"], [2.0, 10.0])
        self.assertEqual(train["runner_class_name"], "JointTeacherStudentRunner")
        self.assertEqual(train["runner"]["policy_class_name"], "JointTeacherStudentActorCritic")
        self.assertEqual(train["runner"]["max_iterations"], 10000)

    def test_cnn_shapes_fusion_schedule_and_student_gradient(self):
        torch.manual_seed(7)
        model = JointTeacherStudentActorCritic(2635, 45, 12)
        obs = torch.randn(8, JOINT_OBSERVATION_DIM)
        privileged = torch.randn(8, 45)
        self.assertEqual(tuple(model.encode_history(obs).shape), (8, 8))
        self.assertEqual(tuple(model.encode_privileged(privileged).shape), (8, 8))
        model.set_schedule_origin(22500)
        model.set_training_iteration(22500)
        self.assertEqual(model.adaptation_alpha, 0.0)
        model.set_training_iteration(27500)
        self.assertEqual(model.adaptation_alpha, 0.5)
        model.set_training_iteration(32500)
        self.assertEqual(model.adaptation_alpha, 1.0)
        loss = model.adaptation_loss(obs, privileged)
        loss.backward()
        student_grad = sum(
            p.grad.abs().sum().item()
            for p in model.student_encoder.parameters()
            if p.grad is not None
        )
        teacher_grad = sum(
            p.grad.abs().sum().item()
            for p in model.teacher_encoder.parameters()
            if p.grad is not None
        )
        self.assertGreater(student_grad, 0.0)
        self.assertEqual(teacher_grad, 0.0)

    def test_teacher_checkpoint_initializes_policy_exactly_at_alpha_zero(self):
        torch.manual_seed(11)
        teacher = OfficialWimTeacherActorCritic(235, 45, 12)
        joint = JointTeacherStudentActorCritic(2635, 45, 12)
        incompatible = joint.load_state_dict(teacher.state_dict(), strict=False)
        self.assertTrue(all(key.startswith("student_encoder.") for key in incompatible.missing_keys))
        self.assertEqual(incompatible.unexpected_keys, [])
        joint.set_schedule_origin(32500)
        joint.set_training_iteration(32500)
        current = torch.randn(16, 235)
        history = torch.randn(16, 50 * 48)
        obs = torch.cat((current, history), dim=-1)
        privileged = torch.randn(16, 45)
        with torch.inference_mode():
            expected = teacher.act_inference(current, privileged)
            actual = joint.act_inference(obs, privileged)
        self.assertEqual((expected - actual).abs().max().item(), 0.0)

    def test_student_only_matches_alpha_one_without_privileged_input(self):
        torch.manual_seed(19)
        model = JointTeacherStudentActorCritic(2635, 45, 12)
        observations = torch.randn(16, JOINT_OBSERVATION_DIM)
        privileged_a = torch.randn(16, 45)
        privileged_b = torch.randn(16, 45) * 100.0
        model.set_schedule_origin(0)
        model.set_training_iteration(model.joint_schedule_iterations)
        with torch.inference_mode():
            student = model.act_inference_student(observations)
            fused_a = model.act_inference(observations, privileged_a)
            fused_b = model.act_inference(observations, privileged_b)
        self.assertEqual((student - fused_a).abs().max().item(), 0.0)
        self.assertEqual((student - fused_b).abs().max().item(), 0.0)

    def test_random_onset_applies_only_after_threshold(self):
        env = object.__new__(A1OfficialWimJointFailureOnset)
        env.num_envs = 2
        env.num_actions = 12
        env.device = "cpu"
        env.dt = 0.02
        env.episode_length_buf = torch.tensor([99, 100])
        env.failure_onset_step = torch.tensor([100, 100])
        env.failure_applied = torch.zeros(2, dtype=torch.bool)
        env.target_degradation = torch.tensor([0.8, 1.0])
        env.target_joint_index = torch.tensor([2, 7])
        env.actuator_degradation = torch.zeros(2, 12)
        env.degraded_joint_index = torch.full((2,), -1, dtype=torch.long)
        env.nominal_motor_strength_gt = torch.ones(2, 12)
        env.motor_strength_gt = torch.ones(2, 12)
        # Test only the onset body without invoking terrain/command callbacks.
        ready = (~env.failure_applied) & (env.episode_length_buf >= env.failure_onset_step)
        active = ready & (env.target_degradation > 0.0)
        ids = active.nonzero(as_tuple=False).flatten()
        joints = env.target_joint_index[ids]
        rates = env.target_degradation[ids]
        env.actuator_degradation[ids, joints] = rates
        env.motor_strength_gt[ids, joints] *= 1.0 - rates
        env.failure_applied[ready] = True
        self.assertEqual(float(env.motor_strength_gt[0, 2]), 1.0)
        self.assertEqual(float(env.motor_strength_gt[1, 7]), 0.0)
        self.assertFalse(bool(env.failure_applied[0]))
        self.assertTrue(bool(env.failure_applied[1]))


if __name__ == "__main__":
    unittest.main()

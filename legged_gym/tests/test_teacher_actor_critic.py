import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import isaacgym  # noqa: F401; Isaac Gym must precede torch
import torch

from legged_gym.learning.teacher_actor_critic import TeacherActorCritic
from legged_gym.learning.teacher_ppo import TeacherPPO
from legged_gym.learning.teacher_runner import TeacherOnPolicyRunner
from rsl_rl.runners import OnPolicyRunner


class _ScalarWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, key, value, iteration):
        self.scalars[key] = (value, iteration)


class TeacherActorCriticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TeacherActorCritic(45, 45, 12)
        self.obs = torch.randn(16, 45)
        self.privileged_obs = torch.randn(16, 45)

    def test_exact_architecture_and_shapes(self):
        self.assertEqual([self.model.teacher_encoder[i].out_features for i in (0, 2, 4, 6)], [512, 256, 128, 8])
        self.assertEqual(self.model.actor[0].in_features, 53)
        self.assertEqual([self.model.actor[i].out_features for i in (0, 2, 4)], [256, 128, 12])
        latent = self.model.encode_privileged(self.privileged_obs)
        actions = self.model.act(self.obs, self.privileged_obs)
        values = self.model.evaluate(self.obs, self.privileged_obs)
        self.assertEqual(tuple(latent.shape), (16, 8))
        self.assertEqual(tuple(actions.shape), (16, 12))
        self.assertEqual(tuple(values.shape), (16, 1))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(self.model.get_actions_log_prob(actions)).all())

    def test_exact_shape_guards(self):
        with self.assertRaises(ValueError):
            TeacherActorCritic(49, 45, 12)
        with self.assertRaises(ValueError):
            TeacherActorCritic(45, 45, 12, teacher_latent_dim=7)
        with self.assertRaises(ValueError):
            TeacherActorCritic(45, 46, 12)
        with self.assertRaises(RuntimeError):
            self.model.act_inference(torch.randn(2, 44), torch.randn(2, 45))

    def test_raw_coordinate_and_storage_clone(self):
        algorithm = TeacherPPO(
            self.model,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            desired_kl=0.01,
            min_learning_rate=1.0e-5,
            max_learning_rate=1.0e-2,
        )
        original_obs = self.obs.clone()
        original_privileged = self.privileged_obs.clone()
        environment_action = algorithm.act(self.obs, self.privileged_obs)
        raw = algorithm.transition.actions
        self.assertTrue(torch.equal(environment_action, raw))
        self.assertTrue(torch.isfinite(algorithm.transition.actions_log_prob).all())
        self.obs.add_(100.0)
        self.privileged_obs.add_(100.0)
        self.assertTrue(torch.equal(algorithm.transition.observations, original_obs))
        self.assertTrue(torch.equal(algorithm.transition.critic_observations, original_privileged))

    def test_raw_inference_mean_is_not_squashed(self):
        final = self.model.actor[-1]
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            final.bias.fill_(2.0)
        mean = self.model.act_inference(self.obs, self.privileged_obs)
        self.assertTrue(torch.equal(mean, torch.full_like(mean, 2.0)))

    def test_runner_serializes_raw_coordinate_provenance(self):
        runner = TeacherOnPolicyRunner.__new__(TeacherOnPolicyRunner)
        runner.env = SimpleNamespace(
            cfg=SimpleNamespace(env=SimpleNamespace(task_name="a1_limping_base_v2")),
            num_obs=45,
            num_envs=4,
        )
        runner.device = "cpu"
        runner.tot_timesteps = 0
        runner.tot_time = 0.0
        runner.num_steps_per_env = 24
        runner.writer = _ScalarWriter()
        runner.alg = SimpleNamespace(
            learning_rate=1.0e-3,
            actor_critic=SimpleNamespace(action_std=torch.ones(1)),
        )
        ppo_metrics = {
            "hard_kl_stopped": 0.0,
            "max_policy_kl": None,
        }
        locs = {
            "it": 3,
            "collection_time": 1.0,
            "learn_time": 1.0,
            "mean_value_loss": 0.0,
            "mean_surrogate_loss": 0.0,
            "rollout_raw_mean_abs": 0.0,
            "rollout_action_abs": 0.0,
            "rollout_deterministic_action_abs": 0.0,
            "rollout_near_bound": 0.0,
            "rollout_forward_velocity": 0.0,
            "rollout_reset_rate": 0.0,
            "rollout_diagnostics": {},
            "ppo_metrics": ppo_metrics,
            "rewbuffer": [],
            "lenbuffer": [],
            "ep_infos": [],
        }
        train_cfg = {
            "runner": {},
            "policy": {},
            "algorithm": {},
        }
        with tempfile.TemporaryDirectory(prefix="teacher-runner-") as temporary:
            runner.log_dir = temporary
            resolved = runner._resolved_config(train_cfg)
            self.assertIn(
                "raw Gaussian",
                resolved["attribution"]["action_coordinate"],
            )
            with mock.patch.object(OnPolicyRunner, "log", return_value=None):
                runner.log(locs)
            metrics = json.loads(
                (Path(temporary) / "metrics.jsonl").read_text(encoding="utf-8")
            )
        self.assertEqual(metrics["PPO/hard_kl_stopped"], 0.0)
        self.assertEqual(runner.writer.scalars["PPO/hard_kl_stopped"], (0.0, 3))

    def test_teacher_receives_actor_and_critic_gradients(self):
        self.model.act(self.obs, self.privileged_obs)
        values = self.model.evaluate(self.obs, self.privileged_obs)
        loss = self.model.action_mean.square().mean() + values.square().mean()
        loss.backward()
        total = sum(
            parameter.grad.abs().sum().item()
            for parameter in self.model.teacher_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(total, 0.0)


if __name__ == "__main__":
    unittest.main()

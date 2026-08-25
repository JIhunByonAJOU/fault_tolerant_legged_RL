import math
import unittest
from unittest import mock

import isaacgym  # noqa: F401; Isaac Gym must precede torch
import torch
import torch.nn as nn

from legged_gym.learning.teacher_ppo import TeacherPPO


class _ControlledActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.location = nn.Parameter(torch.zeros(()))

    def update_distribution(self, obs, privileged_obs):
        batch_size = obs.shape[0]
        self.action_mean = self.location.expand(batch_size, 1)
        self.action_std = (self.location * 0.0 + 1.0).expand(batch_size, 1)
        self.entropy = (self.location * 0.0).expand(batch_size, 1)

    def get_actions_log_prob(self, actions):
        return -0.5 * torch.square(actions - self.action_mean).sum(dim=-1)

    def evaluate(self, obs, privileged_obs):
        return (self.location * 0.0).expand(obs.shape[0], 1)


class _ControlledStorage:
    def __init__(self, batches):
        self.batches = batches
        self.clear_calls = 0

    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        return iter(self.batches)

    def clear(self):
        self.clear_calls += 1


def _batch(old_mean):
    batch_size = 4
    zeros = torch.zeros(batch_size, 1)
    return (
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        torch.zeros(batch_size),
        torch.full((batch_size, 1), old_mean),
        torch.ones(batch_size, 1),
        None,
        None,
    )


def _measured_kl(old_mean):
    batch = _batch(old_mean)
    old_mu = batch[7]
    old_sigma = batch[8]
    mu = torch.zeros_like(old_mu)
    sigma = torch.ones_like(old_sigma)
    kl = torch.sum(
        torch.log(sigma / old_sigma + 1.0e-5)
        + (torch.square(old_sigma) + torch.square(old_mu - mu))
        / (2.0 * torch.square(sigma))
        - 0.5,
        dim=-1,
    )
    return torch.mean(kl).item()


class TeacherPPORawGaussianTest(unittest.TestCase):
    def _algorithm(self, batches):
        algorithm = TeacherPPO(
            _ControlledActorCritic(),
            num_learning_epochs=1,
            num_mini_batches=len(batches),
            schedule="adaptive",
            desired_kl=0.01,
        )
        algorithm.storage = _ControlledStorage(batches)
        return algorithm

    def test_nominal_batch_completes(self):
        algorithm = self._algorithm([_batch(0.0)])
        with mock.patch.object(
            algorithm.optimizer, "step", wraps=algorithm.optimizer.step
        ) as step:
            algorithm.update()
        self.assertEqual(step.call_count, 1)
        self.assertEqual(algorithm.storage.clear_calls, 1)
        self.assertEqual(algorithm.last_update_metrics["completed_updates"], 1)
        self.assertEqual(algorithm.last_update_metrics["planned_updates"], 1)
        self.assertEqual(algorithm.last_update_metrics["hard_kl_stopped"], 0.0)
        self.assertEqual(algorithm.last_update_metrics["early_stopped"], 0.0)
        self.assertIsNone(algorithm.last_update_metrics["max_policy_kl"])

    def test_large_kl_is_diagnostic_and_does_not_skip_update(self):
        old_mean = math.sqrt(0.16)
        algorithm = self._algorithm([_batch(0.0), _batch(old_mean)])
        with mock.patch.object(
            algorithm.optimizer, "step", wraps=algorithm.optimizer.step
        ) as step:
            algorithm.update()
        metrics = algorithm.last_update_metrics
        self.assertEqual(step.call_count, 2)
        self.assertEqual(metrics["completed_updates"], 2)
        self.assertEqual(metrics["planned_updates"], 2)
        self.assertGreater(metrics["max_kl"], 0.05)
        self.assertEqual(metrics["hard_kl_stopped"], 0.0)
        self.assertEqual(metrics["early_stopped"], 0.0)
        self.assertEqual(metrics["nonfinite_update_skipped"], 0.0)
        self.assertIsNone(metrics["max_policy_kl"])
        self.assertEqual(algorithm.storage.clear_calls, 1)


if __name__ == "__main__":
    unittest.main()

"""Checks that onset interventions alter only Student history."""

import unittest
from collections import deque

import isaacgym  # noqa: F401
import torch

from legged_gym.scripts.evaluate_teacher243_failure_onset import (
    condition_for_env,
    delayed_history_at_step,
    intervene_history,
)


class OnsetHistoryInterventionTest(unittest.TestCase):
    def test_actual_zero_and_shuffled(self):
        observations = torch.arange(3 * 2635, dtype=torch.float32).reshape(3, 2635)
        original = observations.clone()
        self.assertIs(intervene_history(observations, "actual"), observations)
        zero = intervene_history(observations, "zero")
        shuffled = intervene_history(observations, "shuffled")
        self.assertTrue(torch.equal(zero[:, :235], original[:, :235]))
        self.assertTrue(torch.all(zero[:, 235:] == 0))
        self.assertTrue(torch.equal(shuffled[:, :235], original[:, :235]))
        self.assertTrue(torch.equal(shuffled[:, 235:], torch.roll(original[:, 235:], 1, 0)))
        cross_joint = intervene_history(observations, "shuffled", shuffle_offset=2)
        self.assertTrue(torch.equal(cross_joint[:, 235:], torch.roll(original[:, 235:], 2, 0)))
        self.assertTrue(torch.equal(observations, original))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            intervene_history(torch.zeros(2, 2635), "unknown")

    def test_full_matrix_shuffle_pairs_other_joint_same_severity_and_onset(self):
        rates = (0.2, 0.4, 0.6, 0.8, 1.0)
        onsets = (2.0, 5.0, 10.0)
        for env_id in range(180):
            joint, rate, onset = condition_for_env(env_id, rates, onsets)
            donor_joint, donor_rate, donor_onset = condition_for_env((env_id - 90) % 180, rates, onsets)
            self.assertEqual((donor_rate, donor_onset), (rate, onset))
            self.assertEqual(donor_joint, (joint + 6) % 12)

    def test_delay_returns_same_robots_earlier_history(self):
        buffer = deque(maxlen=3)
        values = [torch.tensor([[i, i + 10], [i + 20, i + 30]], dtype=torch.float32)
                  for i in range(5)]
        chosen = [delayed_history_at_step(buffer, value, 2).clone() for value in values]
        self.assertTrue(torch.equal(chosen[0], values[0]))
        self.assertTrue(torch.equal(chosen[1], values[1]))
        self.assertTrue(torch.equal(chosen[2], values[0]))
        self.assertTrue(torch.equal(chosen[3], values[1]))
        self.assertTrue(torch.equal(chosen[4], values[2]))


if __name__ == "__main__":
    unittest.main()

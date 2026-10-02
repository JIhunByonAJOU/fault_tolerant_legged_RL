"""Adaptation-only optimizer for the separate Student comparison pipeline."""

import time

import torch
from torch.nn.utils import clip_grad_norm_

from .teacher_ppo import TeacherPPO
from .shared_gpu_pacing import synchronize_active_cuda


class StudentDistillation(TeacherPPO):
    """Train only the Student encoder on on-policy Student rollouts."""

    def __init__(self, actor_critic, learning_rate=1.0e-5, **kwargs):
        super().__init__(actor_critic, learning_rate=learning_rate, **kwargs)
        if self.num_learning_epochs != 5 or self.num_mini_batches != 4:
            raise ValueError("separate Student requires 5 epochs x 4 minibatches")
        if float(learning_rate) != 1.0e-5:
            raise ValueError("separate Student requires fixed learning rate 1e-5")
        if float(self.max_grad_norm) != 1.0:
            raise ValueError("separate Student requires grad clip 1.0")
        self.learning_rate = 1.0e-5
        self.optimizer = torch.optim.Adam(
            self.actor_critic.student_encoder.parameters(), lr=self.learning_rate
        )
        self.last_update_metrics = {}

    def update(self):
        initial_student_parameters = [
            parameter.detach().clone()
            for parameter in self.actor_critic.student_encoder.parameters()
        ]
        adaptation_total = 0.0
        action_mse_total = 0.0
        gradient_norm_total = 0.0
        completed_updates = 0
        minibatch_sync_count = 0
        minibatch_sync_seconds = 0.0
        nonfinite_update_skipped = False
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            obs_batch, privileged_obs_batch = batch[0], batch[1]
            adaptation_loss = self.actor_critic.adaptation_loss(
                obs_batch, privileged_obs_batch
            )
            with torch.no_grad():
                action_mse = self.actor_critic.student_action_loss(
                    obs_batch, privileged_obs_batch
                )
            self.optimizer.zero_grad()
            if not torch.isfinite(adaptation_loss):
                nonfinite_update_skipped = True
                break
            adaptation_loss.backward()
            gradient_norm = clip_grad_norm_(
                self.actor_critic.student_encoder.parameters(), self.max_grad_norm
            )
            if not torch.isfinite(gradient_norm):
                self.optimizer.zero_grad()
                nonfinite_update_skipped = True
                break
            self.optimizer.step()
            if self.shared_gpu_minibatch_sleep_ms > 0.0:
                minibatch_sync_seconds += synchronize_active_cuda(self.device)
                minibatch_sync_count += 1
                time.sleep(self.shared_gpu_minibatch_sleep_ms / 1000.0)
            adaptation_total += adaptation_loss.item()
            action_mse_total += action_mse.item()
            gradient_norm_total += gradient_norm.item()
            completed_updates += 1

        self.storage.clear()
        divisor = max(completed_updates, 1)
        with torch.no_grad():
            parameter_step_l2 = torch.sqrt(
                sum(
                    torch.sum(torch.square(parameter.detach() - initial))
                    for parameter, initial in zip(
                        self.actor_critic.student_encoder.parameters(),
                        initial_student_parameters,
                    )
                )
            ).item()
        self.last_update_metrics = {
            "mode": "adaptation_only",
            "ppo_updates": 0,
            "planned_supervised_updates": self.num_learning_epochs
            * self.num_mini_batches,
            "completed_supervised_updates": completed_updates,
            "adaptation_l2_unsquared": adaptation_total / divisor,
            "action_mse_diagnostic": action_mse_total / divisor,
            "action_mse_weight": 0.0,
            "mean_student_gradient_norm": gradient_norm_total / divisor,
            "student_parameter_step_l2": parameter_step_l2,
            "nonfinite_update_skipped": bool(nonfinite_update_skipped),
            "shared_gpu_minibatch_sleep_ms": self.shared_gpu_minibatch_sleep_ms,
            "minibatch_sync_count": minibatch_sync_count,
            "minibatch_sync_seconds": minibatch_sync_seconds,
        }
        return 0.0, 0.0

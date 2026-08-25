import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO


class TeacherPPO(PPO):
    """PPO variant that passes privileged GT to the teacher actor and critic."""

    def __init__(
        self,
        actor_critic,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-5,
        max_learning_rate=1.0e-2,
        **kwargs
    ):
        super().__init__(
            actor_critic,
            learning_rate=learning_rate,
            **kwargs
        )
        self.min_learning_rate = float(min_learning_rate)
        self.max_learning_rate = float(max_learning_rate)
        self.last_update_metrics = {}

    def act(self, obs, privileged_obs):
        environment_actions, raw_actions = self.actor_critic.act_with_raw(
            obs, privileged_obs
        )
        # Environment, storage, and log probability share the raw Gaussian
        # coordinate; clipping happens only inside the environment.
        self.transition.actions = raw_actions.detach()
        self.transition.values = self.actor_critic.evaluate(
            obs, privileged_obs
        ).detach()
        self.transition.actions_log_prob = (
            self.actor_critic.get_actions_log_prob(self.transition.actions)
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # process_env_step() copies this transition only after env.step().
        # Environment buffers, especially privileged_obs_buf, are updated
        # in-place during that step.  Clone now so the stored state remains the
        # exact o_t/e_t that produced this action, old mean, and log-prob.
        self.transition.observations = obs.clone()
        self.transition.critic_observations = privileged_obs.clone()
        return environment_actions.detach()

    def compute_returns(self, last_obs, last_privileged_obs):
        last_values = self.actor_critic.evaluate(
            last_obs, last_privileged_obs
        ).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_kl = 0.0
        mean_clip_fraction = 0.0
        max_kl = 0.0
        max_abs_log_ratio = 0.0
        mean_gradient_norm = 0.0
        mean_adaptation_loss = 0.0
        completed_updates = 0
        early_stopped = False
        nonfinite_update_skipped = False
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            privileged_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _,
            _,
        ) in generator:
            self.actor_critic.update_distribution(obs_batch, privileged_obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(obs_batch, privileged_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    dim=-1,
                )
                kl_mean = torch.mean(kl)
                kl_value = kl_mean.item()
                mean_kl += kl_value
                max_kl = max(max_kl, kl_value)

                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            self.min_learning_rate,
                            self.learning_rate / 1.5,
                        )
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(
                            self.max_learning_rate,
                            self.learning_rate * 1.5,
                        )
                    for parameter_group in self.optimizer.param_groups:
                        parameter_group["lr"] = self.learning_rate

            log_ratio = actions_log_prob_batch - torch.squeeze(
                old_actions_log_prob_batch
            )
            if not torch.isfinite(log_ratio).all():
                nonfinite_update_skipped = True
                early_stopped = True
                break
            max_abs_log_ratio = max(
                max_abs_log_ratio, log_ratio.detach().abs().max().item()
            )
            ratio = torch.exp(log_ratio)
            mean_clip_fraction += (
                (torch.abs(ratio - 1.0) > self.clip_param)
                .float()
                .mean()
                .item()
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            adaptation_loss = None
            if hasattr(self.actor_critic, "adaptation_loss"):
                adaptation_loss = self.actor_critic.adaptation_loss(
                    obs_batch, privileged_obs_batch
                )
                loss = loss + self.actor_critic.adaptation_beta * adaptation_loss

            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                nonfinite_update_skipped = True
                early_stopped = True
                break
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            if not torch.isfinite(gradient_norm):
                self.optimizer.zero_grad()
                nonfinite_update_skipped = True
                early_stopped = True
                break
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_gradient_norm += gradient_norm.item()
            if adaptation_loss is not None:
                mean_adaptation_loss += adaptation_loss.item()
            completed_updates += 1

        self.storage.clear()
        divisor = max(completed_updates, 1)
        kl_divisor = max(completed_updates + int(early_stopped), 1)
        self.last_update_metrics = {
            "mean_kl": mean_kl / kl_divisor,
            "max_kl": max_kl,
            "mean_clip_fraction": mean_clip_fraction / divisor,
            "max_abs_log_ratio": max_abs_log_ratio,
            "mean_gradient_norm": mean_gradient_norm / divisor,
            "completed_updates": completed_updates,
            "planned_updates": self.num_learning_epochs * self.num_mini_batches,
            "early_stopped": float(early_stopped),
            "hard_kl_stopped": 0.0,
            "max_policy_kl": None,
            "nonfinite_update_skipped": float(nonfinite_update_skipped),
            "adaptation_loss": mean_adaptation_loss / divisor,
            "adaptation_alpha": float(
                getattr(self.actor_critic, "adaptation_alpha", 0.0)
            ),
            "adaptation_beta": float(
                getattr(self.actor_critic, "adaptation_beta", 0.0)
            ),
        }
        return mean_value_loss / divisor, mean_surrogate_loss / divisor

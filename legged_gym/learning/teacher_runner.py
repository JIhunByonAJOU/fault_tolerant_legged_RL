import json
import os
import re
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners import OnPolicyRunner

from legged_gym.envs.a1_limping.a1_limping_config import EXPERIMENT_NAME
from legged_gym.envs.a1_limping.schema import schema_manifest
from legged_gym.utils.helpers import class_to_dict

from .teacher_actor_critic import TeacherActorCritic
from .official_wim_teacher_actor_critic import OfficialWimTeacherActorCritic
from .joint_teacher_student_actor_critic import (
    JointTeacherStudentActorCritic,
    FrozenTeacherStudentActorCritic,
)
from .teacher_ppo import TeacherPPO


class TeacherOnPolicyRunner(OnPolicyRunner):
    """Teacher-only runner that keeps actor observations and GT separate."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is None:
            raise ValueError("TeacherOnPolicyRunner requires privileged observations")
        policy_classes = {
            "TeacherActorCritic": TeacherActorCritic,
            "OfficialWimTeacherActorCritic": OfficialWimTeacherActorCritic,
            "JointTeacherStudentActorCritic": JointTeacherStudentActorCritic,
            "FrozenTeacherStudentActorCritic": FrozenTeacherStudentActorCritic,
        }
        policy_class_name = self.cfg["policy_class_name"]
        if policy_class_name not in policy_classes:
            raise ValueError("Teacher runner received unsupported policy class")
        if self.cfg["algorithm_class_name"] != "TeacherPPO":
            raise ValueError("Teacher runner requires TeacherPPO")

        actor_critic = policy_classes[policy_class_name](
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        self.alg = TeacherPPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._stop_requested = False
        self._stop_reason = None
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "resolved_config.json"), "w") as handle:
                json.dump(
                    self._resolved_config(train_cfg), handle, indent=2, sort_keys=True
                )
        self.env.reset()

    def request_stop(self, reason="requested"):
        """Ask learn() to checkpoint and exit at the next iteration boundary."""
        self._stop_requested = True
        self._stop_reason = str(reason)
        print("Graceful training stop requested: {}".format(self._stop_reason), flush=True)

    def _resolved_config(self, train_cfg):
        """Return the provenance-explicit baseline/deviation contract."""
        runner_cfg = getattr(self, "cfg", train_cfg["runner"])
        policy_class_name = runner_cfg.get("policy_class_name")
        if policy_class_name in (
            "JointTeacherStudentActorCritic",
            "FrozenTeacherStudentActorCritic",
        ):
            from legged_gym.envs.a1_official_wim_teacher.joint_schema import (
                schema_manifest as joint_schema_manifest,
            )
            frozen_teacher = policy_class_name == "FrozenTeacherStudentActorCritic"
            beta_floor = float(self.policy_cfg.get("adaptation_beta_floor", 0.0))
            if frozen_teacher:
                schedule_description = (
                    "teacher-only {} iterations, then alpha 0->1 over {} iterations; "
                    "beta=1; Teacher encoder, actor, and action std frozen"
                ).format(
                    self.policy_cfg["teacher_only_iterations"],
                    self.policy_cfg["student_transition_iterations"],
                )
            else:
                schedule_description = (
                    "linear alpha 0->1 and beta 1->{:.3g} over configured JT iterations"
                ).format(beta_floor)
            return {
                "task": getattr(self.env.cfg.env, "task_name", "unknown"),
                "mode": (
                    "frozen_teacher_student_random_onset"
                    if frozen_teacher else "joint_teacher_student_random_onset"
                ),
                "experiment_name": runner_cfg["experiment_name"],
                "attribution": {
                    "teacher": "selected [TF] warm-full checkpoint",
                    "student": "Saving 1-D CNN history50 latent8; RMA-lineage CNN layout",
                    "fusion": "Saving alpha*z_student + (1-alpha)*z_teacher",
                    "implementation_choices": [
                        "history uses first 48 WIM observation values",
                        schedule_description,
                        "random degradation onset uniformly sampled from 2 to 10 seconds",
                    ],
                },
                "schema": joint_schema_manifest(),
                "environment": class_to_dict(self.env.cfg),
                "training": train_cfg,
            }
        if runner_cfg.get("policy_class_name") == "OfficialWimTeacherActorCritic":
            from legged_gym.envs.a1_official_wim_teacher.schema import (
                schema_manifest as teacher243_schema_manifest,
            )

            task_name = getattr(self.env.cfg.env, "task_name", "unknown")
            declared_deviations = [
                "nonduplicate privileged45 physical context",
                "actor and critic input expanded 235 -> 243",
                "mild payload, motor-strength, Kp, and Kd randomization",
            ]
            if task_name == "a1_official_wim_teacher243_failure":
                declared_deviations.append(
                    "one episode-constant actuator degradation at d in {0.2,0.4,0.6,0.8}, with intact episodes retained"
                )
            return {
                "task": task_name,
                "mode": "official_wim_obs235_plus_teacher_latent8",
                "experiment_name": runner_cfg["experiment_name"],
                "attribution": {
                    "baseline": "successful official WIM A1 rough obs235 policy",
                    "teacher_encoder": "Saving Paper Explicit MLP [512,256,128] -> latent8",
                    "declared_deviations": declared_deviations,
                    "preserved": [
                        "official WIM terrain observations 187",
                        "official WIM rewards, commands, PPO, action coordinate, and A1 control",
                    ],
                },
                "schema": teacher243_schema_manifest(),
                "environment": class_to_dict(self.env.cfg),
                "training": train_cfg,
            }
        return {
            "task": getattr(self.env.cfg.env, "task_name", "unknown"),
            "mode": "privileged_teacher_only",
            "experiment_name": EXPERIMENT_NAME,
            "attribution": {
                "baseline": "Official initial WIM/A1 code experiment",
                "privileged_teacher_architecture": "Saving Paper Explicit: teacher45 -> latent8",
                "declared_deviations": [
                    "plane actor45 observation layout",
                    "nonduplicate privileged45 hidden-factor layout",
                    "active payload, motor-strength, Kp-scale, and Kd-scale randomization",
                    "actor and critic receive actor45 plus teacher latent8",
                    "diagnostics and bounded validation budgets",
                ],
                "numerical_safety": [
                    "finite update guard",
                    "rollout storage clone",
                ],
                "action_coordinate": "raw Gaussian sample, storage, log probability, and inference mean; environment-only clip",
            },
            "official_baseline": {
                "action_scale": 0.25,
                "entropy_coef": 0.01,
                "only_positive_rewards": True,
                "num_steps_per_env": 24,
                "num_mini_batches": 4,
                "num_learning_epochs": 5,
                "desired_kl": 0.01,
                "local_hard_kl_stop": False,
            },
            "schema": schema_manifest(self.env.num_obs),
            "environment": class_to_dict(self.env.cfg),
            "training": train_cfg,
            "validation_budgets": {
                "p0": {"num_envs": 64, "iterations": 2, "transition_cap": 3072},
                "conditional_p1": {"num_envs": 4096, "iterations": 1500},
            },
        }

    def save(self, path, infos=None, iteration=None):
        """Save the global iteration represented by a periodic checkpoint.

        The upstream runner names periodic files with the loop iteration but
        serializes ``current_learning_iteration``, which remains at the run's
        starting value until ``learn`` returns.  That silently rewinds the
        reward curriculum when a periodic teacher checkpoint is resumed.
        """
        checkpoint_iteration = (
            self.current_learning_iteration if iteration is None else iteration
        )
        temporary_path = "{}.tmp-{}".format(path, os.getpid())
        try:
            torch.save(
                {
                    "model_state_dict": self.alg.actor_critic.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "iter": checkpoint_iteration,
                    "infos": infos,
                },
                temporary_path,
            )
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        if loaded.get("warm_start_from_official_wim"):
            self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
            if load_optimizer and "optimizer_state_dict" in loaded:
                self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
                continued_lr = float(self.alg.optimizer.param_groups[0]["lr"])
                self.alg.learning_rate = continued_lr
            self.current_learning_iteration = int(loaded.get("iter", 0))
            return loaded.get("infos")
        infos = super().load(path, load_optimizer=load_optimizer)
        match = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
        if match is not None:
            filename_iteration = int(match.group(1))
            if self.current_learning_iteration != filename_iteration:
                print(
                    "Correcting checkpoint iteration {} -> {} from filename".format(
                        self.current_learning_iteration, filename_iteration
                    )
                )
                self.current_learning_iteration = filename_iteration
        return infos

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        privileged_obs = self.env.get_privileged_observations().to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        total_iterations = self.current_learning_iteration + num_learning_iterations
        completed_iterations = 0
        for iteration in range(self.current_learning_iteration, total_iterations):
            if hasattr(self.alg.actor_critic, "set_training_iteration"):
                self.alg.actor_critic.set_training_iteration(iteration)
            if hasattr(self.env, "set_training_iteration"):
                self.env.set_training_iteration(iteration)
            start = time.time()
            rollout_raw_mean_abs = torch.zeros((), device=self.device)
            rollout_action_abs = torch.zeros((), device=self.device)
            rollout_deterministic_action_abs = torch.zeros((), device=self.device)
            rollout_near_bound = torch.zeros((), device=self.device)
            rollout_forward_velocity = torch.zeros((), device=self.device)
            rollout_reset_rate = torch.zeros((), device=self.device)
            rollout_diagnostic_sums = {}
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)
                    raw_mean = self.alg.actor_critic.action_mean
                    rollout_raw_mean_abs += raw_mean.abs().mean()
                    rollout_action_abs += actions.abs().mean()
                    rollout_deterministic_action_abs += raw_mean.abs().mean()
                    rollout_near_bound += (
                        torch.clip(actions, -1.0, 1.0).abs() >= 0.98
                    ).float().mean()
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    rollout_forward_velocity += self.env.base_lin_vel[:, 0].mean()
                    rollout_reset_rate += (dones > 0).float().mean()
                    if hasattr(self.env, "get_rollout_diagnostics"):
                        for key, value in self.env.get_rollout_diagnostics().items():
                            value = torch.as_tensor(value, device=self.device)
                            if key not in rollout_diagnostic_sums:
                                rollout_diagnostic_sums[key] = torch.zeros(
                                    (), device=self.device
                                )
                            rollout_diagnostic_sums[key] += value.float().mean()
                    obs = obs.to(self.device)
                    privileged_obs = privileged_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                latent_diagnostics = {}
                actor_critic = self.alg.actor_critic
                if hasattr(actor_critic, "act_inference_with_latent"):
                    latent = actor_critic.encode_privileged(privileged_obs)
                    latent_std = latent.std(dim=0, unbiased=False)
                    actual_mean = actor_critic.act_inference_with_latent(obs, latent)
                    zero_mean = actor_critic.act_inference_with_latent(
                        obs, torch.zeros_like(latent)
                    )
                    shuffled_mean = actor_critic.act_inference_with_latent(
                        obs, torch.roll(latent, shifts=1, dims=0)
                    )
                    latent_diagnostics = {
                        "mean_abs": latent.abs().mean().item(),
                        "mean_std": latent_std.mean().item(),
                        "noncollapsed_dimensions": int(
                            (latent_std > 1.0e-3).sum().item()
                        ),
                        "action_delta_zero": (
                            actual_mean - zero_mean
                        ).abs().mean().item(),
                        "action_delta_shuffled": (
                            actual_mean - shuffled_mean
                        ).abs().mean().item(),
                    }
                    if hasattr(actor_critic, "encode_history"):
                        student_latent = actor_critic.encode_history(obs)
                        student_mean = actor_critic.act_inference_with_latent(
                            obs, student_latent
                        )
                        latent_diagnostics.update(
                            {
                                "student_teacher_l2": torch.linalg.vector_norm(
                                    student_latent - latent, dim=-1
                                ).mean().item(),
                                "student_teacher_action_delta": (
                                    student_mean - actual_mean
                                ).abs().mean().item(),
                            }
                        )
                learn_start = time.time()
                self.alg.compute_returns(obs, privileged_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            learn_time = time.time() - learn_start
            if self.log_dir is not None:
                self.log(
                    {
                        "it": iteration,
                        "num_learning_iterations": num_learning_iterations,
                        "collection_time": collection_time,
                        "learn_time": learn_time,
                        "ep_infos": ep_infos,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "mean_value_loss": mean_value_loss,
                        "mean_surrogate_loss": mean_surrogate_loss,
                        "ppo_metrics": dict(self.alg.last_update_metrics),
                        "rollout_raw_mean_abs": (
                            rollout_raw_mean_abs / self.num_steps_per_env
                        ).item(),
                        "rollout_action_abs": (
                            rollout_action_abs / self.num_steps_per_env
                        ).item(),
                        "rollout_deterministic_action_abs": (
                            rollout_deterministic_action_abs
                            / self.num_steps_per_env
                        ).item(),
                        "rollout_near_bound": (
                            rollout_near_bound / self.num_steps_per_env
                        ).item(),
                        "rollout_forward_velocity": (
                            rollout_forward_velocity / self.num_steps_per_env
                        ).item(),
                        "rollout_reset_rate": (
                            rollout_reset_rate / self.num_steps_per_env
                        ).item(),
                        "rollout_diagnostics": {
                            key: (value / self.num_steps_per_env).item()
                            for key, value in rollout_diagnostic_sums.items()
                        },
                        "latent_diagnostics": latent_diagnostics,
                    }
                )
            if iteration % self.save_interval == 0 and self.log_dir is not None:
                self.save(
                    os.path.join(self.log_dir, "model_{}.pt".format(iteration)),
                    iteration=iteration,
                )
            ep_infos.clear()
            completed_iterations += 1
            if self._stop_requested:
                print(
                    "Stopping at completed iteration {}: {}".format(
                        iteration, self._stop_reason
                    ),
                    flush=True,
                )
                break

        self.current_learning_iteration += completed_iterations
        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir,
                    "model_{}.pt".format(self.current_learning_iteration),
                )
            )
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
                self.writer = None

    def log(self, locs, width=80, pad=35):
        """Keep TensorBoard, JSONL, and W&B on one compact metric row."""
        super().log(locs, width=width, pad=pad)

        rollout_metrics = {
            "Policy/raw_mean_abs": locs["rollout_raw_mean_abs"],
            "Policy/sampled_action_abs": locs["rollout_action_abs"],
            "Policy/deterministic_action_abs": locs[
                "rollout_deterministic_action_abs"
            ],
            "Policy/action_near_bound_rate": locs["rollout_near_bound"],
            "Rollout/mean_forward_velocity": locs["rollout_forward_velocity"],
            "Rollout/reset_rate_per_step": locs["rollout_reset_rate"],
        }
        for key, value in locs.get("rollout_diagnostics", {}).items():
            if key.endswith("_mse"):
                metric_name = key[: -len("_mse")] + "_rmse"
                value = max(value, 0.0) ** 0.5
            else:
                metric_name = key
            rollout_metrics["Rollout/" + metric_name] = value
        ppo_metrics = {
            "PPO/mean_kl": locs["ppo_metrics"].get("mean_kl", 0.0),
            "PPO/max_kl": locs["ppo_metrics"].get("max_kl", 0.0),
            "PPO/mean_clip_fraction": locs["ppo_metrics"].get(
                "mean_clip_fraction", 0.0
            ),
            "PPO/max_abs_log_ratio": locs["ppo_metrics"].get(
                "max_abs_log_ratio", 0.0
            ),
            "PPO/mean_gradient_norm": locs["ppo_metrics"].get(
                "mean_gradient_norm", 0.0
            ),
            "PPO/completed_updates": locs["ppo_metrics"].get(
                "completed_updates", 0
            ),
            "PPO/planned_updates": locs["ppo_metrics"].get(
                "planned_updates", 0
            ),
            "PPO/early_stopped": locs["ppo_metrics"].get(
                "early_stopped", 0.0
            ),
            "PPO/hard_kl_stopped": locs["ppo_metrics"].get(
                "hard_kl_stopped", 0.0
            ),
            "PPO/max_policy_kl": locs["ppo_metrics"].get("max_policy_kl"),
            "PPO/nonfinite_update_skipped": locs["ppo_metrics"].get(
                "nonfinite_update_skipped", 0.0
            ),
            "Adaptation/loss": locs["ppo_metrics"].get("adaptation_loss"),
            "Adaptation/alpha": locs["ppo_metrics"].get("adaptation_alpha"),
            "Adaptation/beta": locs["ppo_metrics"].get("adaptation_beta"),
        }
        latent_metrics = {
            "Latent/mean_abs": locs.get("latent_diagnostics", {}).get(
                "mean_abs"
            ),
            "Latent/mean_std": locs.get("latent_diagnostics", {}).get(
                "mean_std"
            ),
            "Latent/noncollapsed_dimensions": locs.get(
                "latent_diagnostics", {}
            ).get("noncollapsed_dimensions"),
            "Latent/action_delta_zero": locs.get("latent_diagnostics", {}).get(
                "action_delta_zero"
            ),
            "Latent/action_delta_shuffled": locs.get(
                "latent_diagnostics", {}
            ).get("action_delta_shuffled"),
            "Adaptation/student_teacher_l2": locs.get(
                "latent_diagnostics", {}
            ).get("student_teacher_l2"),
            "Adaptation/student_teacher_action_delta": locs.get(
                "latent_diagnostics", {}
            ).get("student_teacher_action_delta"),
        }
        for key, value in {
            **rollout_metrics,
            **ppo_metrics,
            **latent_metrics,
        }.items():
            if value is not None:
                self.writer.add_scalar(key, value, locs["it"])

        iteration_time = locs["collection_time"] + locs["learn_time"]
        metrics = {
            "iteration": locs["it"],
            "wall_time_unix": time.time(),
            "total_transitions": self.tot_timesteps,
            "Loss/value_function": locs["mean_value_loss"],
            "Loss/surrogate": locs["mean_surrogate_loss"],
            "Loss/learning_rate": self.alg.learning_rate,
            "Policy/mean_noise_std": self.alg.actor_critic.action_std.mean().item(),
            "Perf/total_fps": int(
                self.num_steps_per_env * self.env.num_envs / iteration_time
            ),
            "Perf/collection_time": locs["collection_time"],
            "Perf/learning_time": locs["learn_time"],
            "Perf/iteration_time": iteration_time,
            "Perf/elapsed_time": self.tot_time,
            **rollout_metrics,
            **{key: value for key, value in ppo_metrics.items() if value is not None},
            **{key: value for key, value in latent_metrics.items() if value is not None},
        }
        if len(locs["rewbuffer"]) > 0:
            metrics["Train/mean_reward"] = statistics.mean(locs["rewbuffer"])
            metrics["Train/mean_episode_length"] = statistics.mean(
                locs["lenbuffer"]
            )

        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                values = []
                for ep_info in locs["ep_infos"]:
                    value = ep_info[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.as_tensor(value, device=self.device)
                    values.append(value.reshape(-1).to(self.device))
                metrics["Episode/" + key] = torch.cat(values).mean().item()

        metrics_path = os.path.join(self.log_dir, "metrics.jsonl")
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            handle.flush()

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        wandb.log(metrics, step=locs["it"])

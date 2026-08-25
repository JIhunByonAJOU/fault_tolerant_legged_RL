"""Random-onset FullRange environment for joint teacher-student training."""

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot

from .a1_official_wim_teacher import A1OfficialWimTeacher243
from .a1_official_wim_teacher_failure import A1OfficialWimTeacher243Failure
from .joint_schema import (
    CURRENT_OBSERVATION_DIM,
    HISTORY_FRAME_DIM,
    HISTORY_LENGTH,
    JOINT_OBSERVATION_DIM,
)


class A1OfficialWimJointFailureOnset(A1OfficialWimTeacher243Failure):
    """Starts intact, then degrades one actuator at a sampled episode time."""

    def _init_buffers(self):
        # Bypass the episode-start sampler in the FailureEnv parent.
        LeggedRobot._init_buffers(self)
        self._init_teacher_context(validate_observation_dim=False)
        self.nominal_motor_strength_gt = self.motor_strength_gt.clone()
        self.actuator_degradation = torch.zeros_like(self.motor_strength_gt)
        self.degraded_joint_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.target_degradation = torch.zeros(
            self.num_envs, dtype=self.motor_strength_gt.dtype, device=self.device
        )
        self.target_joint_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.failure_onset_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.failure_applied = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.observation_history = torch.zeros(
            self.num_envs,
            HISTORY_LENGTH,
            HISTORY_FRAME_DIM,
            dtype=self.obs_buf.dtype,
            device=self.device,
        )
        self._sample_onset_failure(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )

    def _sample_onset_failure(self, env_ids):
        if len(env_ids) == 0:
            return
        self.nominal_motor_strength_gt[env_ids] = self.motor_strength_gt[env_ids]
        self.actuator_degradation[env_ids] = 0.0
        self.motor_strength_gt[env_ids] = self.nominal_motor_strength_gt[env_ids]
        self.degraded_joint_index[env_ids] = -1
        self.failure_applied[env_ids] = False

        levels = torch.as_tensor(
            self.cfg.domain_rand.actuator_degradation_levels,
            dtype=self.motor_strength_gt.dtype,
            device=self.device,
        )
        sampled = torch.randint(levels.numel(), (len(env_ids),), device=self.device)
        self.target_degradation[env_ids] = levels[sampled]
        self.target_joint_index[env_ids] = torch.randint(
            self.num_actions, (len(env_ids),), device=self.device
        )
        low_s, high_s = self.cfg.domain_rand.failure_onset_time_range_s
        low_step = int(round(float(low_s) / self.dt))
        high_step = int(round(float(high_s) / self.dt))
        self.failure_onset_step[env_ids] = torch.randint(
            low_step, high_step + 1, (len(env_ids),), device=self.device
        )
        self.observation_history[env_ids] = 0.0

    def reset_idx(self, env_ids):
        # Reset mechanics and physical DR without invoking episode-start failure.
        A1OfficialWimTeacher243.reset_idx(self, env_ids)
        if hasattr(self, "actuator_degradation"):
            self._sample_onset_failure(env_ids)

    def _post_physics_step_callback(self):
        LeggedRobot._post_physics_step_callback(self)
        ready = (~self.failure_applied) & (
            self.episode_length_buf >= self.failure_onset_step
        )
        active = ready & (self.target_degradation > 0.0)
        env_ids = active.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            joints = self.target_joint_index[env_ids]
            rates = self.target_degradation[env_ids]
            self.actuator_degradation[env_ids, joints] = rates
            self.degraded_joint_index[env_ids] = joints
            self.motor_strength_gt[env_ids] = self.nominal_motor_strength_gt[env_ids]
            self.motor_strength_gt[env_ids, joints] *= 1.0 - rates
        self.failure_applied[ready] = True

    def compute_observations(self):
        # Reproduce the stock WIM 235-D current observation, then append a
        # reset-safe 50x48 history used only by the student encoder.
        current = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1,
            ) * self.obs_scales.height_measurements
            current = torch.cat((current, heights), dim=-1)
        if current.shape[-1] != CURRENT_OBSERVATION_DIM:
            raise RuntimeError("JT requires current WIM observation235")
        if self.add_noise:
            current = current + (2 * torch.rand_like(current) - 1) * self.noise_scale_vec[
                :CURRENT_OBSERVATION_DIM
            ]

        self.observation_history[:, :-1] = self.observation_history[:, 1:].clone()
        self.observation_history[:, -1] = current[:, :HISTORY_FRAME_DIM]
        self.obs_buf = torch.cat(
            (current, self.observation_history.flatten(start_dim=1)), dim=-1
        )
        if self.obs_buf.shape[-1] != JOINT_OBSERVATION_DIM:
            raise RuntimeError("JT observation width changed")

        failure_flag = (self.actuator_degradation.max(dim=1).values > 0.0).to(
            self.privileged_obs_buf.dtype
        )
        privileged = torch.cat(
            (
                self.applied_friction,
                self.added_base_mass,
                self.motor_strength_gt,
                self.kp_scale_gt,
                self.kd_scale_gt,
                self.base_lin_vel,
                self.base_ang_vel,
                failure_flag.unsqueeze(1),
            ),
            dim=-1,
        )
        self.privileged_obs_buf.copy_(privileged)

    def get_rollout_diagnostics(self):
        values = super().get_rollout_diagnostics()
        values.update(
            {
                "failure_applied": self.failure_applied.float(),
                "failure_onset_seconds": self.failure_onset_step.float() * self.dt,
            }
        )
        return values

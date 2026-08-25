"""Official WIM rough task with a separate, nonduplicated teacher context."""

import numpy as np
import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot

from .schema import (
    PRIVILEGED_OBSERVATION_DIM,
    WIM_OBSERVATION_DIM,
)


class A1OfficialWimTeacher243(LeggedRobot):
    """Keep WIM obs235 unchanged and expose hidden context to the teacher."""

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            self.added_base_mass = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )
        if self.cfg.domain_rand.randomize_base_mass:
            low, high = self.cfg.domain_rand.added_mass_range
            added_mass = float(np.random.uniform(low, high))
            props[0].mass += added_mass
            self.added_base_mass[env_id, 0] = added_mass
        return props

    def _init_buffers(self):
        super()._init_buffers()
        self._init_teacher_context(validate_observation_dim=True)

    def _init_teacher_context(self, validate_observation_dim):
        """Initialize privileged actuator context after base buffers exist."""
        if validate_observation_dim and self.num_obs != WIM_OBSERVATION_DIM:
            raise RuntimeError("teacher243 requires the unchanged WIM obs235")
        if self.num_privileged_obs != PRIVILEGED_OBSERVATION_DIM:
            raise RuntimeError("teacher243 requires privileged45")

        if hasattr(self, "friction_coeffs"):
            self.applied_friction = self.friction_coeffs.reshape(
                self.num_envs, -1
            )[:, :1].to(self.device)
        else:
            self.applied_friction = torch.full(
                (self.num_envs, 1),
                float(self.cfg.terrain.static_friction),
                dtype=torch.float,
                device=self.device,
            )
        if not hasattr(self, "added_base_mass"):
            self.added_base_mass = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )

        shape = (self.num_envs, self.num_actions)
        self.motor_strength_gt = torch.ones(shape, device=self.device)
        self.kp_scale_gt = torch.ones(shape, device=self.device)
        self.kd_scale_gt = torch.ones(shape, device=self.device)
        self._sample_actuator_context(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )

    def _sample_actuator_context(self, env_ids):
        if len(env_ids) == 0:
            return
        shape = (len(env_ids), self.num_actions)
        cfg = self.cfg.domain_rand
        self.motor_strength_gt[env_ids] = torch_rand_float(
            *cfg.motor_strength_range, shape, device=self.device
        )
        self.kp_scale_gt[env_ids] = torch_rand_float(
            *cfg.kp_scale_range, shape, device=self.device
        )
        self.kd_scale_gt[env_ids] = torch_rand_float(
            *cfg.kd_scale_range, shape, device=self.device
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._sample_actuator_context(env_ids)

    def compute_observations(self):
        # This call preserves the official WIM 48+187 observation exactly.
        super().compute_observations()
        failure_flag = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
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
                failure_flag,
            ),
            dim=-1,
        )
        if privileged.shape != (self.num_envs, PRIVILEGED_OBSERVATION_DIM):
            raise RuntimeError("teacher243 privileged observation schema changed")
        self.privileged_obs_buf.copy_(privileged)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        if self.cfg.control.control_type != "P":
            raise NameError("teacher243 requires official WIM P control")
        torques = (
            self.p_gains
            * self.kp_scale_gt
            * (actions_scaled + self.default_dof_pos - self.dof_pos)
            - self.d_gains * self.kd_scale_gt * self.dof_vel
        )
        torques *= self.motor_strength_gt
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import get_euler_xyz, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi

from .schema import (
    ACTOR_OBSERVATION_DIM,
    PRIVILEGED_OBSERVATION_DIM,
    ActorObservationSlices,
    PrivilegedObservationSlices,
)


class A1LimpingBase(LeggedRobot):
    """WIM-flat A1 environment with a Saving-style privileged teacher."""

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            self.added_base_mass = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )
        if self.cfg.domain_rand.randomize_base_mass:
            mass_range = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(mass_range[0], mass_range[1])
            props[0].mass += added_mass
            self.added_base_mass[env_id, 0] = added_mass
        return props

    def _init_buffers(self):
        super()._init_buffers()
        if self.num_obs != ACTOR_OBSERVATION_DIM:
            raise RuntimeError(
                "WIM-flat A1 requires {} actor observations, got {}".format(
                    ACTOR_OBSERVATION_DIM, self.num_obs
                )
            )

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, -1, 13
        )
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
        else:
            self.added_base_mass = self.added_base_mass.to(self.device)

        shape = (self.num_envs, self.num_actions)
        self.motor_strength_gt = torch.ones(shape, dtype=torch.float, device=self.device)
        self.kp_scale_gt = torch.ones(shape, dtype=torch.float, device=self.device)
        self.kd_scale_gt = torch.ones(shape, dtype=torch.float, device=self.device)
        self.raw_actions = torch.zeros(shape, dtype=torch.float, device=self.device)
        self._sample_actuator_randomization(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )
        self.privileged_obs_raw_buf = torch.zeros_like(self.privileged_obs_buf)

    def step(self, actions):
        """Preserve the raw Gaussian action and let the base environment clip it."""
        self.raw_actions.copy_(actions.to(self.device))
        return super().step(actions)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.raw_actions[env_ids] = 0.0
            self._sample_actuator_randomization(env_ids)

    def _sample_actuator_randomization(self, env_ids):
        if len(env_ids) == 0:
            return
        shape = (len(env_ids), self.num_actions)
        domain_rand = self.cfg.domain_rand
        self.motor_strength_gt[env_ids] = torch_rand_float(
            *domain_rand.motor_strength_range, shape, device=self.device
        )
        self.kp_scale_gt[env_ids] = torch_rand_float(
            *domain_rand.kp_scale_range, shape, device=self.device
        )
        self.kd_scale_gt[env_ids] = torch_rand_float(
            *domain_rand.kd_scale_range, shape, device=self.device
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        scales = self.cfg.noise.noise_scales
        level = self.cfg.noise.noise_level
        noise_vec[ActorObservationSlices.DOF_POSITION] = (
            scales.dof_pos * level * self.obs_scales.dof_pos
        )
        noise_vec[ActorObservationSlices.DOF_VELOCITY] = (
            scales.dof_vel * level * self.obs_scales.dof_vel
        )
        return noise_vec

    def compute_observations(self):
        roll, pitch, _ = get_euler_xyz(self.base_quat)
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.privileged.foot_contact_force_threshold
        ).float()
        self.obs_buf = torch.cat(
            (
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=-1),
                contacts,
                self.raw_actions,
                self.commands[:, :3] * self.commands_scale,
            ),
            dim=-1,
        )
        if self.obs_buf.shape[1] != ACTOR_OBSERVATION_DIM:
            raise RuntimeError(
                "Actor observation schema mismatch: expected {}, got {}".format(
                    ACTOR_OBSERVATION_DIM, self.obs_buf.shape[1]
                )
            )
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        self._compute_privileged_observations()

    def _compute_privileged_observations(self):
        failure_flag = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        raw = torch.cat(
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
        if raw.shape[1] != PRIVILEGED_OBSERVATION_DIM:
            raise RuntimeError(
                "Privileged observation schema mismatch: expected {}, got {}".format(
                    PRIVILEGED_OBSERVATION_DIM, raw.shape[1]
                )
            )
        self.privileged_obs_raw_buf.copy_(raw)

        self.privileged_obs_buf.copy_(raw)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        if self.cfg.control.control_type != "P":
            raise NameError("Plane teacher BaseEnv requires P control")
        torques = (
            self.p_gains
            * self.kp_scale_gt
            * (actions_scaled + self.default_dof_pos - self.dof_pos)
            - self.d_gains * self.kd_scale_gt * self.dof_vel
        )
        torques *= self.motor_strength_gt
        return torch.clip(torques, -self.torque_limits, self.torque_limits)


class A1LimpingBaseV2(A1LimpingBase):
    """The stable V2 task key plus non-rewarding rollout diagnostics."""

    def _init_buffers(self):
        super()._init_buffers()
        zeros = lambda: torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.episode_metric_steps = zeros()
        self.episode_path_length = zeros()
        self.episode_yaw_change = zeros()
        self.episode_vx_error_sq = zeros()
        self.episode_vy_error_sq = zeros()
        self.episode_yaw_error_sq = zeros()
        self.episode_vertical_velocity_sq = zeros()
        self.episode_ground_impact = zeros()
        self.episode_foot_slip = zeros()
        self.episode_diagonal_contacts = zeros()
        self.episode_four_feet_contacts = zeros()
        self.episode_airborne = zeros()
        self.episode_start_xy = self.root_states[:, :2].clone()
        self.episode_previous_xy = self.root_states[:, :2].clone()
        _, _, yaw = get_euler_xyz(self.base_quat)
        self.episode_previous_yaw = wrap_to_pi(yaw).clone()
        base_height = self._diagnostic_base_height()
        self.episode_min_base_height = base_height.clone()
        self.episode_max_base_height = base_height.clone()
        self.previous_diagnostic_contact_forces = self.contact_forces[
            :, self.feet_indices, :
        ].clone()
        self._rollout_diagnostics = {}

    def post_physics_step(self):
        super().post_physics_step()
        self.previous_diagnostic_contact_forces.copy_(
            self.contact_forces[:, self.feet_indices, :]
        )
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self.previous_diagnostic_contact_forces[reset_ids] = 0.0

    def _diagnostic_base_height(self):
        if isinstance(self.measured_heights, torch.Tensor) and self.measured_heights.ndim == 2:
            terrain_height = torch.mean(self.measured_heights, dim=1)
        else:
            terrain_height = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        return self.root_states[:, 2] - terrain_height

    def _contact_pattern_masks(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.privileged.foot_contact_force_threshold
        )
        count = contacts.sum(dim=1)
        diagonal = (count == 2) & (
            (contacts[:, 0] & contacts[:, 3])
            | (contacts[:, 1] & contacts[:, 2])
        )
        return diagonal, count == 4, count == 0

    def _diagnostic_ground_impact(self):
        delta = (
            self.contact_forces[:, self.feet_indices, :]
            - self.previous_diagnostic_contact_forces
        )
        return torch.sum(torch.square(delta), dim=(1, 2))

    def _diagnostic_foot_slip(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.privileged.foot_contact_force_threshold
        ).float()
        velocity_xy = self.rigid_body_states[:, self.feet_indices, 7:9]
        return torch.sum(
            contacts * torch.sum(torch.square(velocity_xy), dim=-1), dim=1
        )

    def _update_episode_diagnostics(self):
        current_xy = self.root_states[:, :2]
        self.episode_path_length += torch.norm(
            current_xy - self.episode_previous_xy, dim=1
        )
        self.episode_previous_xy.copy_(current_xy)
        _, _, yaw = get_euler_xyz(self.base_quat)
        yaw = wrap_to_pi(yaw)
        self.episode_yaw_change += wrap_to_pi(yaw - self.episode_previous_yaw)
        self.episode_previous_yaw.copy_(yaw)

        vx_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        vy_error = self.base_lin_vel[:, 1] - self.commands[:, 1]
        yaw_error = self.base_ang_vel[:, 2] - self.commands[:, 2]
        vertical_velocity_sq = torch.square(self.base_lin_vel[:, 2])
        impact = self._diagnostic_ground_impact()
        slip = self._diagnostic_foot_slip()
        diagonal, four_feet, airborne = self._contact_pattern_masks()
        base_height = self._diagnostic_base_height()

        self.episode_metric_steps += 1.0
        self.episode_vx_error_sq += torch.square(vx_error)
        self.episode_vy_error_sq += torch.square(vy_error)
        self.episode_yaw_error_sq += torch.square(yaw_error)
        self.episode_vertical_velocity_sq += vertical_velocity_sq
        self.episode_ground_impact += impact
        self.episode_foot_slip += slip
        self.episode_diagonal_contacts += diagonal.float()
        self.episode_four_feet_contacts += four_feet.float()
        self.episode_airborne += airborne.float()
        self.episode_min_base_height = torch.minimum(self.episode_min_base_height, base_height)
        self.episode_max_base_height = torch.maximum(self.episode_max_base_height, base_height)
        self._rollout_diagnostics = {
            "command_vx_mse": torch.square(vx_error),
            "command_vy_mse": torch.square(vy_error),
            "command_yaw_rate_mse": torch.square(yaw_error),
            "vertical_velocity_mse": vertical_velocity_sq,
            "mean_base_height": base_height,
            "ground_impact": impact,
            "foot_slip": slip,
            "diagonal_contact_rate": diagonal.float(),
            "four_feet_contact_rate": four_feet.float(),
            "airborne_rate": airborne.float(),
        }

    def get_rollout_diagnostics(self):
        return self._rollout_diagnostics

    def compute_reward(self):
        self._update_episode_diagnostics()
        super().compute_reward()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        valid_ids = env_ids[self.episode_metric_steps[env_ids] > 0]
        episode_metrics = None
        if len(valid_ids) > 0:
            steps = self.episode_metric_steps[valid_ids].clamp_min(1.0)
            displacement = torch.norm(
                self.root_states[valid_ids, :2] - self.episode_start_xy[valid_ids], dim=1
            )
            episode_metrics = {
                "command_vx_rmse": torch.sqrt(self.episode_vx_error_sq[valid_ids] / steps).mean(),
                "command_vy_rmse": torch.sqrt(self.episode_vy_error_sq[valid_ids] / steps).mean(),
                "command_yaw_rate_rmse": torch.sqrt(self.episode_yaw_error_sq[valid_ids] / steps).mean(),
                "world_net_displacement": displacement.mean(),
                "path_efficiency": (displacement / self.episode_path_length[valid_ids].clamp_min(1.0e-6)).mean(),
                "abs_net_yaw_change": torch.abs(self.episode_yaw_change[valid_ids]).mean(),
                "vertical_velocity_rms": torch.sqrt(self.episode_vertical_velocity_sq[valid_ids] / steps).mean(),
                "base_height_peak_to_peak": (self.episode_max_base_height[valid_ids] - self.episode_min_base_height[valid_ids]).mean(),
                "mean_ground_impact": (self.episode_ground_impact[valid_ids] / steps).mean(),
                "mean_foot_slip": (self.episode_foot_slip[valid_ids] / steps).mean(),
                "diagonal_contact_rate": (self.episode_diagonal_contacts[valid_ids] / steps).mean(),
                "four_feet_contact_rate": (self.episode_four_feet_contacts[valid_ids] / steps).mean(),
                "airborne_rate": (self.episode_airborne[valid_ids] / steps).mean(),
            }
            full_horizon = steps >= 0.95 * float(self.max_episode_length)
            eligible = (~self.time_out_buf[valid_ids]) | full_horizon
            if torch.any(eligible):
                episode_metrics["survival_rate_20s"] = self.time_out_buf[
                    valid_ids[eligible]
                ].float().mean()
                episode_metrics["survival_rate_20s_valid"] = torch.ones(
                    (), dtype=torch.float, device=self.device
                )
            else:
                episode_metrics["survival_rate_20s"] = torch.zeros(
                    (), dtype=torch.float, device=self.device
                )
                episode_metrics["survival_rate_20s_valid"] = torch.zeros(
                    (), dtype=torch.float, device=self.device
                )

        super().reset_idx(env_ids)
        if episode_metrics is not None:
            self.extras["episode"].update(episode_metrics)

        for buffer in (
            self.episode_metric_steps,
            self.episode_path_length,
            self.episode_yaw_change,
            self.episode_vx_error_sq,
            self.episode_vy_error_sq,
            self.episode_yaw_error_sq,
            self.episode_vertical_velocity_sq,
            self.episode_ground_impact,
            self.episode_foot_slip,
            self.episode_diagonal_contacts,
            self.episode_four_feet_contacts,
            self.episode_airborne,
        ):
            buffer[env_ids] = 0.0
        self.episode_start_xy[env_ids] = self.root_states[env_ids, :2]
        self.episode_previous_xy[env_ids] = self.root_states[env_ids, :2]
        _, _, yaw = get_euler_xyz(self.base_quat[env_ids])
        self.episode_previous_yaw[env_ids] = wrap_to_pi(yaw)
        base_height = self._diagnostic_base_height()[env_ids]
        self.episode_min_base_height[env_ids] = base_height
        self.episode_max_base_height[env_ids] = base_height
        self.previous_diagnostic_contact_forces[env_ids] = 0.0

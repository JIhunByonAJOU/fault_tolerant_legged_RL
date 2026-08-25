"""Teacher243 environment with one episode-constant degraded actuator."""

import torch

from .a1_official_wim_teacher import A1OfficialWimTeacher243


class A1OfficialWimTeacher243Failure(A1OfficialWimTeacher243):
    """Expose actuator location and severity through the existing motor-strength context.

    The ADAPT convention is used: degradation ``d=0`` is intact and ``d=1``
    removes all actuator output.  At most one of the twelve entries is nonzero.
    """

    def _init_buffers(self):
        super()._init_buffers()
        self.nominal_motor_strength_gt = self.motor_strength_gt.clone()
        self.actuator_degradation = torch.zeros_like(self.motor_strength_gt)
        self.degraded_joint_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._sample_actuator_degradation(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )

    def _sample_actuator_degradation(self, env_ids):
        if len(env_ids) == 0:
            return
        self.nominal_motor_strength_gt[env_ids] = self.motor_strength_gt[env_ids]
        self.actuator_degradation[env_ids] = 0.0
        self.degraded_joint_index[env_ids] = -1

        levels = torch.as_tensor(
            self.cfg.domain_rand.actuator_degradation_levels,
            dtype=self.motor_strength_gt.dtype,
            device=self.device,
        )
        level_indices = torch.randint(
            levels.numel(), (len(env_ids),), device=self.device
        )
        sampled_levels = levels[level_indices]
        sampled_joints = torch.randint(
            self.num_actions, (len(env_ids),), device=self.device
        )
        failed = sampled_levels > 0.0
        failed_env_ids = env_ids[failed]
        failed_joints = sampled_joints[failed]
        self.actuator_degradation[failed_env_ids, failed_joints] = sampled_levels[failed]
        self.degraded_joint_index[failed_env_ids] = failed_joints
        self.motor_strength_gt[env_ids] = self.nominal_motor_strength_gt[env_ids] * (
            1.0 - self.actuator_degradation[env_ids]
        )

    def set_actuator_degradation(self, env_ids, joint_index, degradation_rate):
        """Set an exact evaluation condition without changing tensor dimensions."""
        if not 0 <= int(joint_index) < self.num_actions:
            raise ValueError("joint_index must be in [0, 11]")
        rate = float(degradation_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("degradation_rate must be in [0, 1]")
        self.actuator_degradation[env_ids] = 0.0
        self.actuator_degradation[env_ids, int(joint_index)] = rate
        self.degraded_joint_index[env_ids] = int(joint_index) if rate > 0.0 else -1
        self.motor_strength_gt[env_ids] = self.nominal_motor_strength_gt[env_ids] * (
            1.0 - self.actuator_degradation[env_ids]
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "actuator_degradation"):
            self._sample_actuator_degradation(env_ids)

    def compute_observations(self):
        super().compute_observations()
        # The motor-strength slice already carries joint identity and severity.
        # This scalar only distinguishes intact from degraded episodes.
        self.privileged_obs_buf[:, -1] = (
            self.actuator_degradation.max(dim=1).values > 0.0
        ).to(self.privileged_obs_buf.dtype)

    def get_rollout_diagnostics(self):
        parent = {}
        parent_getter = getattr(super(), "get_rollout_diagnostics", None)
        if parent_getter is not None:
            parent = dict(parent_getter())
        active = self.actuator_degradation.max(dim=1).values
        parent.update(
            {
                "degradation_rate": active,
                "degraded_episode": (active > 0.0).float(),
                "effective_motor_strength_min": self.motor_strength_gt.min(dim=1).values,
            }
        )
        return parent

"""64-env GPU PhysX integration for the WIM/Saving teacher baseline."""

import os
import subprocess
import sys

import isaacgym  # noqa: F401; Isaac Gym must precede torch
import pytest
import torch

from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.envs.a1_limping.schema import (
    ACTOR_OBSERVATION_DIM,
    PRIVILEGED_OBSERVATION_DIM,
    ActorObservationSlices,
    PrivilegedObservationSlices,
    schema_manifest,
)
from legged_gym.learning.teacher_actor_critic import TeacherActorCritic
from legged_gym.learning.teacher_ppo import TeacherPPO
from legged_gym.utils import get_args, task_registry


EXPECTED_REWARDS = {
    "tracking_lin_vel": 0.02,
    "tracking_ang_vel": 0.01,
    "lin_vel_z": -0.04,
    "ang_vel_xy": -0.001,
    "dof_acc": -5e-9,
    "torques": -2e-7,
    "action_rate": -0.0002,
    "collision": -0.02,
    "feet_air_time": 0.02,
}


class _NoGraphicsGymProxy:
    forbidden = {
        "create_viewer", "step_graphics", "draw_viewer",
        "sync_frame_time", "poll_viewer_events",
    }

    def __init__(self, target):
        self.target = target
        self.calls = []

    def __getattr__(self, name):
        attribute = getattr(self.target, name)
        if name not in self.forbidden:
            return attribute

        def counted(*args, **kwargs):
            self.calls.append(name)
            return attribute(*args, **kwargs)

        return counted


@pytest.fixture
def args(monkeypatch):
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "--headless"])
    return get_args()


def _assert_resolved_contract(env_cfg, train_cfg):
    assert env_cfg.env.num_envs == 4096
    assert env_cfg.env.num_observations == 45
    assert env_cfg.env.num_privileged_obs == 45
    assert env_cfg.terrain.mesh_type == "plane"
    assert env_cfg.terrain.curriculum is False
    assert env_cfg.commands.num_commands == 4
    assert env_cfg.commands.heading_command is True
    assert env_cfg.commands.resampling_time == 10.0
    assert env_cfg.commands.ranges.lin_vel_x == [-1.0, 1.0]
    assert env_cfg.commands.ranges.lin_vel_y == [-1.0, 1.0]
    assert env_cfg.domain_rand.friction_range == [0.5, 1.25]
    assert env_cfg.domain_rand.push_interval_s == 10.0
    assert env_cfg.domain_rand.max_push_vel_xy == 1.0
    assert env_cfg.domain_rand.randomize_base_mass is True
    assert env_cfg.domain_rand.added_mass_range == [0.0, 6.0]
    assert env_cfg.domain_rand.motor_strength_range == [0.9, 1.1]
    assert env_cfg.noise.add_noise is True
    assert env_cfg.noise.noise_scales.lin_vel == 0.01
    assert env_cfg.control.action_scale == 0.25
    assert env_cfg.control.decimation == 4
    assert env_cfg.control.stiffness == {"joint": 20.0}
    assert env_cfg.control.damping == {"joint": 0.5}
    assert env_cfg.sim.dt == 0.005
    assert env_cfg.asset.terminate_after_contacts_on == ["base"]
    assert env_cfg.rewards.only_positive_rewards is True
    assert env_cfg.rewards.tracking_sigma == 0.25
    assert train_cfg.runner.experiment_name == "Official initial WIM A1 + plane Saving teacher45 baseline"
    assert train_cfg.runner.num_steps_per_env == 24
    assert train_cfg.runner.max_iterations == 1500
    assert train_cfg.algorithm.num_mini_batches == 4
    assert train_cfg.algorithm.num_learning_epochs == 5
    assert train_cfg.algorithm.learning_rate == 1.0e-3
    assert train_cfg.algorithm.min_learning_rate == 1.0e-5
    assert train_cfg.algorithm.max_learning_rate == 1.0e-2
    assert train_cfg.algorithm.entropy_coef == 0.01
    assert train_cfg.algorithm.gamma == 0.99
    assert train_cfg.algorithm.lam == 0.95
    assert train_cfg.algorithm.desired_kl == 0.01
    assert not hasattr(train_cfg.algorithm, "max_policy_kl")


def _run_1000_step_smoke(task_name, args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    _assert_resolved_contract(env_cfg, train_cfg)
    env_cfg.env.num_envs = 64
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    assert args.headless
    assert env.graphics_device_id == -1
    assert env.viewer is None
    assert env.device == "cuda:0"
    assert "legged_gym.evaluation.gait_metrics" not in sys.modules
    env.gym = _NoGraphicsGymProxy(env.gym)

    obs, privileged_obs = env.reset()
    assert obs.shape == (64, ACTOR_OBSERVATION_DIM)
    assert privileged_obs.shape == (64, PRIVILEGED_OBSERVATION_DIM)
    assert schema_manifest()["actor_input_dim"] == 53
    assert schema_manifest()["critic_input_dim"] == 53
    assert torch.count_nonzero(
        env.privileged_obs_raw_buf[:, PrivilegedObservationSlices.FAILURE_FLAG]
    ) == 0
    factor_contract = (
        (PrivilegedObservationSlices.FRICTION, env.applied_friction, [0.5, 1.25]),
        (PrivilegedObservationSlices.ADDED_BASE_MASS, env.added_base_mass, [0.0, 6.0]),
        (PrivilegedObservationSlices.MOTOR_STRENGTH, env.motor_strength_gt, [0.9, 1.1]),
        (PrivilegedObservationSlices.KP_SCALE, env.kp_scale_gt, [50.0 / 55.0, 60.0 / 55.0]),
        (PrivilegedObservationSlices.KD_SCALE, env.kd_scale_gt, [0.4 / 0.6, 0.8 / 0.6]),
    )
    for location, applied, bounds in factor_contract:
        values = env.privileged_obs_raw_buf[:, location]
        assert torch.equal(values, applied)
        assert torch.all((values >= bounds[0]) & (values <= bounds[1]))
        assert torch.any(torch.var(values, dim=0, unbiased=False) > 0.0)
    before = env.privileged_obs_raw_buf[0].clone()
    env.motor_strength_gt[0, 0] += 0.001
    env._compute_privileged_observations()
    changed = torch.nonzero(
        env.privileged_obs_raw_buf[0] != before, as_tuple=False
    ).flatten()
    assert torch.equal(changed, torch.tensor([2], device=env.device))
    env.motor_strength_gt[0, 0] -= 0.001
    env._compute_privileged_observations()
    assert set(env.reward_names) == set(EXPECTED_REWARDS)
    for name, scale in EXPECTED_REWARDS.items():
        assert abs(env.reward_scales[name] - scale) <= max(
            1.0e-12, abs(scale) * 1.0e-6
        ), (name, env.reward_scales[name], scale)
    planar_norm = torch.norm(env.commands[:, :2], dim=1)
    assert torch.all((planar_norm == 0.0) | (planar_norm > 0.2))

    model = TeacherActorCritic(45, 45, 12).to(env.device)
    algorithm = TeacherPPO(
        model,
        device=env.device,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,
        min_learning_rate=1.0e-5,
        max_learning_rate=1.0e-2,
    )
    smoke_actions = algorithm.act(obs, privileged_obs)
    assert smoke_actions.shape == (64, 12)
    assert torch.isfinite(smoke_actions).all()
    assert torch.isfinite(algorithm.transition.actions_log_prob).all()

    test_actions = torch.full((64, 12), 0.1, device=env.device)
    for step in range(1000):
        actions = test_actions if step == 0 else torch.zeros_like(test_actions)
        obs, privileged_obs, rewards, dones, _ = env.step(actions)
        if step == 0:
            assert torch.allclose(obs[:, ActorObservationSlices.PREVIOUS_ACTION], actions)
        contacts = obs[:, ActorObservationSlices.FOOT_CONTACTS]
        assert torch.all((contacts == 0.0) | (contacts == 1.0))
        assert obs.shape == (64, 45)
        assert privileged_obs.shape == (64, 45)
        assert rewards.shape == (64,)
        assert dones.shape == (64,)
        assert torch.isfinite(obs).all()
        assert torch.isfinite(privileged_obs).all()
        assert torch.isfinite(rewards).all()
    assert env.gym.calls == []
    env.gym.destroy_sim(env.sim)


def test_a1_limping_base(args):
    if os.environ.get("LEGGED_GYM_GPU_TEST_CHILD") == "1":
        _run_1000_step_smoke("a1_limping_base", args)
    else:
        environment = os.environ.copy()
        environment["LEGGED_GYM_GPU_TEST_CHILD"] = "1"
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-s", "{}::test_a1_limping_base".format(__file__)],
            cwd=str(__file__.rsplit("/legged_gym/tests/", 1)[0]),
            env=environment,
            check=True,
        )
    print("a1_limping_base 64-env 1000-step GPU smoke passed")


def test_a1_limping_base_v2(args):
    if os.environ.get("LEGGED_GYM_GPU_TEST_CHILD") == "1":
        _run_1000_step_smoke("a1_limping_base_v2", args)
    else:
        environment = os.environ.copy()
        environment["LEGGED_GYM_GPU_TEST_CHILD"] = "1"
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-s", "{}::test_a1_limping_base_v2".format(__file__)],
            cwd=str(__file__.rsplit("/legged_gym/tests/", 1)[0]),
            env=environment,
            check=True,
        )
    print("a1_limping_base_v2 64-env 1000-step GPU smoke passed")


if __name__ == "__main__":
    parsed_args = get_args()
    test_a1_limping_base(parsed_args)
    test_a1_limping_base_v2(parsed_args)

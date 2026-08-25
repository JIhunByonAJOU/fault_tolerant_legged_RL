"""Non-training 64-environment GPU verifier for plane teacher BaseEnv tasks."""

import argparse
import subprocess
import sys

import isaacgym  # noqa: F401; Isaac Gym must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.envs.a1_limping.schema import (
    ACTOR_OBSERVATION_DIM,
    PRIVILEGED_OBSERVATION_DIM,
    ActorObservationSlices,
    PrivilegedObservationSlices,
)
from legged_gym.utils import get_args, task_registry


def _range_and_sensitivity(name, values, bounds):
    if not torch.all((values >= bounds[0]) & (values <= bounds[1])):
        raise AssertionError("{} is outside {}".format(name, bounds))
    if not torch.any(torch.var(values.float(), dim=0, unbiased=False) > 0.0):
        raise AssertionError("{} is constant across environments".format(name))


def _assert_applied_factors(env, privileged):
    blocks = (
        ("friction", PrivilegedObservationSlices.FRICTION, env.applied_friction,
         env.cfg.domain_rand.friction_range),
        ("payload", PrivilegedObservationSlices.ADDED_BASE_MASS, env.added_base_mass,
         env.cfg.domain_rand.added_mass_range),
        ("motor", PrivilegedObservationSlices.MOTOR_STRENGTH, env.motor_strength_gt,
         env.cfg.domain_rand.motor_strength_range),
        ("kp", PrivilegedObservationSlices.KP_SCALE, env.kp_scale_gt,
         env.cfg.domain_rand.kp_scale_range),
        ("kd", PrivilegedObservationSlices.KD_SCALE, env.kd_scale_gt,
         env.cfg.domain_rand.kd_scale_range),
    )
    for name, location, applied, bounds in blocks:
        actual = privileged[:, location]
        if not torch.equal(actual, applied):
            raise AssertionError("{} privileged values differ from applied tensor".format(name))
        _range_and_sensitivity(name, actual, bounds)
    failure = privileged[:, PrivilegedObservationSlices.FAILURE_FLAG]
    if torch.count_nonzero(failure).item() != 0:
        raise AssertionError("reserved failure flag is nonzero")


def _run(task, args, num_envs, steps):
    env_cfg, _ = task_registry.get_cfgs(name=task)
    env_cfg.env.num_envs = num_envs
    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    try:
        obs, privileged = env.reset()
        _assert_applied_factors(env, privileged)
        for step in range(steps):
            raw = torch.zeros(num_envs, env.num_actions, device=env.device)
            if step == 0:
                raw[:, 0] = 1.5
            obs, privileged, rewards, dones, _ = env.step(raw)
            if obs.shape != (num_envs, ACTOR_OBSERVATION_DIM):
                raise AssertionError("actor shape drift: {}".format(tuple(obs.shape)))
            if privileged.shape != (num_envs, PRIVILEGED_OBSERVATION_DIM):
                raise AssertionError("privileged shape drift: {}".format(tuple(privileged.shape)))
            if not all(torch.isfinite(value).all() for value in (obs, privileged, rewards)):
                raise AssertionError("nonfinite tensor at step {}".format(step))
            contacts = obs[:, ActorObservationSlices.FOOT_CONTACTS]
            if not torch.all((contacts == 0.0) | (contacts == 1.0)):
                raise AssertionError("nonbinary contacts at step {}".format(step))
            if step == 0 and not torch.all(obs[:, ActorObservationSlices.PREVIOUS_ACTION][:, 0] == 1.5):
                raise AssertionError("previous action is not the unclipped raw action")
            _assert_applied_factors(env, privileged)
        print("{}: PASS envs={} steps={} obs={} privileged={}".format(
            task, num_envs, steps, tuple(obs.shape), tuple(privileged.shape)
        ))
    finally:
        env.gym.destroy_sim(env.sim)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tasks", default="a1_limping_base,a1_limping_base_v2")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    known, remaining = parser.parse_known_args()
    tasks = [task.strip() for task in known.tasks.split(",") if task.strip()]
    if len(tasks) > 1:
        for task in tasks:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--tasks", task,
                    "--num-envs", str(known.num_envs),
                    "--steps", str(known.steps),
                    "--seed", str(known.seed),
                    "--sim-device", known.sim_device,
                    "--rl-device", known.rl_device,
                    *(["--headless"] if known.headless else []),
                    *remaining,
                ],
                check=True,
            )
        return
    sys.argv = [
        sys.argv[0],
        "--seed", str(known.seed),
        "--sim_device", known.sim_device,
        "--rl_device", known.rl_device,
        *(["--headless"] if known.headless else []),
        *remaining,
    ]
    args = get_args()
    _run(tasks[0], args, known.num_envs, known.steps)


if __name__ == "__main__":
    main()

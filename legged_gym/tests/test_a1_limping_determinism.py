"""Repeat a short fixed-seed BaseEnv trace in two independent simulations."""

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
import torch

from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.utils import get_args, task_registry


@pytest.fixture
def args(monkeypatch):
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "--headless"])
    return get_args()


def capture_trace(args):
    env_cfg, _ = task_registry.get_cfgs(name="a1_limping_base")
    env_cfg.env.num_envs = 16
    env, _ = task_registry.make_env(
        name="a1_limping_base", args=args, env_cfg=env_cfg
    )
    obs, privileged_obs = env.reset()
    trace = [obs.cpu(), privileged_obs.cpu()]

    action_template = torch.linspace(
        -0.2, 0.2, env.num_actions, device=env.device
    ).repeat(env.num_envs, 1)
    for step in range(10):
        actions = action_template * ((step + 1) / 10.0)
        obs, privileged_obs, rewards, dones, _ = env.step(actions)
        trace.extend(
            (
                obs.cpu(),
                privileged_obs.cpu(),
                rewards.unsqueeze(1).cpu(),
                dones.unsqueeze(1).float().cpu(),
            )
        )

    result = torch.cat([item.reshape(-1) for item in trace])
    env.gym.destroy_sim(env.sim)
    return result


def test_determinism(args):
    traces = []
    with tempfile.TemporaryDirectory(prefix="a1-determinism-") as temporary:
        for index in range(2):
            path = Path(temporary) / "trace-{}.pt".format(index)
            subprocess.run(
                [sys.executable, __file__, "--capture", str(path), "--headless"],
                check=True,
            )
            traces.append(torch.load(path))
    first, second = traces
    if not torch.equal(first, second):
        max_error = torch.max(torch.abs(first - second)).item()
        raise AssertionError(
            "Fixed-seed traces differ; max absolute error={}".format(max_error)
        )
    print("a1_limping_base fixed-seed trace is bitwise deterministic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--capture")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if known.capture:
        torch.save(capture_trace(get_args()), known.capture)
    else:
        test_determinism(get_args())

"""Measure whether a trained teacher243 policy actually uses its latent."""

import argparse
import json
import math
import sys
from pathlib import Path

import isaacgym  # noqa: F401; Isaac Gym must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.utils import get_args, task_registry


def parse_args():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--checkpoint-path", required=True)
    own.add_argument("--latent-mode", choices=("actual", "zero", "shuffled"), required=True)
    own.add_argument("--steps", type=int, default=1000)
    own.add_argument("--output", required=True)
    known, remaining = own.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def main():
    own, args = parse_args()
    if args.task != "a1_official_wim_teacher243" or not args.headless:
        raise ValueError("latent evaluation requires teacher243 --headless")
    if own.steps <= 0:
        raise ValueError("--steps must be positive")
    checkpoint = Path(own.checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env, _ = task_registry.make_env(name=args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(checkpoint), load_optimizer=False)
    model = runner.alg.actor_critic
    model.eval()
    obs = env.get_observations()
    privileged = env.get_privileged_observations()

    reward_sum = 0.0
    reset_sum = 0.0
    tracking_error_sq = 0.0
    action_delta_sum = 0.0
    latent_sum = torch.zeros(8, device=env.device)
    latent_sq_sum = torch.zeros(8, device=env.device)
    samples = 0
    with torch.inference_mode():
        for _ in range(own.steps):
            latent = model.encode_privileged(privileged)
            actual_actions = model.act_inference_with_latent(obs, latent)
            if own.latent_mode == "actual":
                used_latent = latent
            elif own.latent_mode == "zero":
                used_latent = torch.zeros_like(latent)
            else:
                used_latent = torch.roll(latent, shifts=1, dims=0)
            actions = model.act_inference_with_latent(obs, used_latent)
            action_delta_sum += (actions - actual_actions).abs().mean().item()
            obs, privileged, rewards, dones, _ = env.step(actions)
            error = env.commands[:, :2] - env.base_lin_vel[:, :2]
            tracking_error_sq += error.square().sum(dim=1).mean().item()
            reward_sum += rewards.mean().item()
            reset_sum += (dones > 0).float().mean().item()
            latent_sum += latent.sum(dim=0)
            latent_sq_sum += latent.square().sum(dim=0)
            samples += latent.shape[0]

    mean = latent_sum / samples
    variance = torch.clamp(latent_sq_sum / samples - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    result = {
        "checkpoint": str(checkpoint),
        "latent_mode": own.latent_mode,
        "num_envs": env.num_envs,
        "steps": own.steps,
        "mean_reward_per_step": reward_sum / own.steps,
        "reset_rate_per_step": reset_sum / own.steps,
        "command_xy_rmse": math.sqrt(tracking_error_sq / own.steps),
        "mean_abs_action_change_vs_actual_latent": action_delta_sum / own.steps,
        "latent_mean": mean.cpu().tolist(),
        "latent_std": std.cpu().tolist(),
        "noncollapsed_latent_dimensions_std_gt_1e-3": int((std > 1.0e-3).sum().item()),
    }
    output = Path(own.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()

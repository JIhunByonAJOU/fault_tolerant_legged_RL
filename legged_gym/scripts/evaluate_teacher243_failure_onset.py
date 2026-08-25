"""Evaluate a trained FailureEnv teacher under sudden mid-episode degradation."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401; Isaac Gym must precede torch
import torch
from isaacgym import gymtorch

from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.scripts.evaluate_official_wim_checkpoint import (
    _acquire_rigid_body_states,
    _refresh_rigid_body_states,
)
from legged_gym.utils import get_args, task_registry


NUM_JOINTS = 12


def parse_args():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--checkpoint-path", required=True)
    own.add_argument("--onset-seconds", type=float, nargs="+", default=(2.0, 5.0, 10.0))
    own.add_argument("--post-seconds", type=float, default=20.0)
    own.add_argument("--rates", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0))
    own.add_argument("--replicates-per-condition", type=int, default=16)
    own.add_argument("--command-x", type=float, default=0.5)
    own.add_argument("--policy-mode", choices=("teacher", "student"), default="teacher")
    own.add_argument("--recovery-window-seconds", type=float, default=1.0)
    own.add_argument("--recovery-vx-error", type=float, default=0.1)
    own.add_argument("--recovery-yaw-rate", type=float, default=0.2)
    own.add_argument("--output", required=True)
    known, remaining = own.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def condition_for_env(env_id, rates, onsets):
    condition = int(env_id) % (NUM_JOINTS * len(rates) * len(onsets))
    onset_index = condition % len(onsets)
    rate_index = (condition // len(onsets)) % len(rates)
    joint_index = condition // (len(onsets) * len(rates))
    return joint_index, rates[rate_index], onsets[onset_index]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path, metadata, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"record_type": "metadata", **metadata}, sort_keys=True) + "\n")
        for row in rows:
            stream.write(json.dumps({"record_type": "robot", **row}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def main():
    own, args = parse_args()
    if args.task not in (
        "a1_official_wim_teacher243_failure_fullrange",
        "a1_official_wim_jt_failure_fullrange_onset",
    ) or not args.headless:
        raise ValueError("onset evaluation requires fullrange failure task with --headless")
    rates = tuple(float(value) for value in own.rates)
    onsets = tuple(float(value) for value in own.onset_seconds)
    if not rates or any(value <= 0.0 or value > 1.0 for value in rates):
        raise ValueError("rates must be unique values in (0, 1]")
    if len(set(rates)) != len(rates):
        raise ValueError("rates must be unique")
    if not onsets or any(value <= 0.0 for value in onsets):
        raise ValueError("onset seconds must be positive")
    if len(set(onsets)) != len(onsets):
        raise ValueError("onset seconds must be unique")
    if own.post_seconds <= 0.0 or own.replicates_per_condition <= 0:
        raise ValueError("post seconds and replicates must be positive")
    checkpoint = Path(own.checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    conditions = NUM_JOINTS * len(rates) * len(onsets)
    num_envs = conditions * own.replicates_per_condition
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = max(onsets) + own.post_seconds + 1.0
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    if args.task == "a1_official_wim_jt_failure_fullrange_onset":
        env_cfg.domain_rand.failure_onset_time_range_s = [1000000.0, 1000000.0]
    env, _ = task_registry.make_env(args.task, args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(checkpoint), load_optimizer=False)
    model = runner.alg.actor_critic
    model.eval()
    rigid_body_states = _acquire_rigid_body_states(env, gymtorch, torch)

    dt = float(env.dt)
    post_steps = int(round(own.post_seconds / dt))
    recovery_window_steps = max(1, int(round(own.recovery_window_seconds / dt)))
    env_ids = torch.arange(num_envs, device=env.device)
    specs = [condition_for_env(i, rates, onsets) for i in range(num_envs)]
    joint_by_env = torch.tensor([x[0] for x in specs], dtype=torch.long, device=env.device)
    rate_by_env = torch.tensor([x[1] for x in specs], dtype=torch.float, device=env.device)
    onset_step_by_env = torch.tensor(
        [int(round(x[2] / dt)) for x in specs], dtype=torch.long, device=env.device
    )
    end_step_by_env = onset_step_by_env + post_steps

    groups = []
    for joint in range(NUM_JOINTS):
        for rate in rates:
            ids = env_ids[(joint_by_env == joint) & torch.isclose(rate_by_env, torch.tensor(rate, device=env.device))]
            groups.append((ids, joint, rate))
    # The FailureEnv normally samples degradation at reset.  This evaluation
    # explicitly starts every robot intact and introduces the assigned failure
    # only when its onset time is reached.
    env.set_actuator_degradation(env_ids, 0, 0.0)

    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    failed_before_onset = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    first_done_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
    post_count = torch.zeros(num_envs, device=env.device)
    reward_sum = torch.zeros(num_envs, device=env.device)
    vx_error_sq = torch.zeros(num_envs, device=env.device)
    yaw_error_sq = torch.zeros(num_envs, device=env.device)
    vertical_sq = torch.zeros(num_envs, device=env.device)
    forward_progress = torch.zeros(num_envs, device=env.device)
    action_near_bound = torch.zeros(num_envs, device=env.device)
    four_contact = torch.zeros(num_envs, device=env.device)
    slip = torch.zeros(num_envs, device=env.device)
    stable_run = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    recovery_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
    max_step = int(end_step_by_env.max().item())

    obs = privileged = None
    if own.policy_mode == "student":
        env.commands[:, :] = 0.0
        env.commands[:, 0] = own.command_x
        env.compute_observations()
        obs = env.get_observations()
        privileged = env.get_privileged_observations()

    with torch.inference_mode():
        for step in range(max_step):
            for ids, joint, rate in groups:
                due = ids[onset_step_by_env[ids] <= step]
                if len(due):
                    env.set_actuator_degradation(due, joint, rate)
            env.commands[:, :] = 0.0
            env.commands[:, 0] = own.command_x
            if own.policy_mode == "teacher":
                env.compute_observations()
                obs = env.get_observations()
                privileged = env.get_privileged_observations()
            if own.policy_mode == "student":
                actions = model.act_inference_student(obs)
            elif hasattr(model, "encode_privileged") and hasattr(
                model, "act_inference_with_latent"
            ):
                # A completed JT checkpoint stores adaptation_alpha=1.  Calling
                # act_inference would therefore select the Student latent even
                # in this Teacher reference mode.  Bypass the schedule and use
                # the privileged encoder explicitly, as the fixed evaluator does.
                teacher_latent = model.encode_privileged(privileged)
                actions = model.act_inference_with_latent(obs, teacher_latent)
            else:
                actions = model.act_inference(obs, privileged)

            post = active & (step >= onset_step_by_env) & (step < end_step_by_env)
            mask = post.float()
            contacts = env.contact_forces[:, env.feet_indices, 2] > 1.0
            foot_velocity = _refresh_rigid_body_states(env, rigid_body_states, torch, "onset step {}".format(step))
            foot_speed = torch.norm(foot_velocity[:, :, :2], dim=-1)
            vx_error = (env.base_lin_vel[:, 0] - own.command_x).abs()
            yaw_rate = env.base_ang_vel[:, 2].abs()
            post_count += mask
            vx_error_sq += vx_error.square() * mask
            yaw_error_sq += yaw_rate.square() * mask
            vertical_sq += env.base_lin_vel[:, 2].square() * mask
            forward_progress += env.base_lin_vel[:, 0] * dt * mask
            action_near_bound += (actions.abs() >= 0.98).float().mean(dim=1) * mask
            four_contact += (contacts.sum(dim=1) == 4).float() * mask
            slip += ((foot_speed * contacts.float()).sum(dim=1) / contacts.float().sum(dim=1).clamp(min=1.0)) * mask

            stable_now = post & (vx_error <= own.recovery_vx_error) & (yaw_rate <= own.recovery_yaw_rate)
            stable_run = torch.where(stable_now, stable_run + 1, torch.zeros_like(stable_run))
            newly_recovered = (recovery_step < 0) & (stable_run >= recovery_window_steps)
            recovery_step[newly_recovered] = step + 1 - recovery_window_steps

            next_obs, next_privileged, rewards, dones, _ = env.step(actions)
            if own.policy_mode == "student":
                obs, privileged = next_obs, next_privileged
            reward_sum += rewards * mask
            newly_done = active & (dones > 0)
            failed_before_onset |= newly_done & (step < onset_step_by_env)
            first_done_step[newly_done] = step + 1
            active[newly_done] = False
            completed = active & ((step + 1) >= end_step_by_env)
            active[completed] = False
            if not active.any():
                break

    denom = post_count.clamp(min=1.0)
    terrain_levels = getattr(env, "terrain_levels", torch.full((num_envs,), -1, device=env.device))
    terrain_types = getattr(env, "terrain_types", torch.full((num_envs,), -1, device=env.device))
    rows = []
    for i, (joint, rate, onset) in enumerate(specs):
        done_step = int(first_done_step[i].item())
        survived = done_step < 0 or done_step >= int(end_step_by_env[i].item())
        recovered = int(recovery_step[i].item()) >= 0
        recovery_seconds = (
            max(0.0, (int(recovery_step[i].item()) - int(onset_step_by_env[i].item())) * dt)
            if recovered else None
        )
        n = float(denom[i].item())
        rows.append({
            "seed": int(args.seed),
            "env_id": i,
            "replicate": i // conditions,
            "joint_index": joint,
            "joint_name": env.dof_names[joint],
            "degradation_rate": rate,
            "remaining_output_fraction": 1.0 - rate,
            "onset_seconds": onset,
            "post_horizon_seconds": own.post_seconds,
            "terrain_level": int(terrain_levels[i].item()),
            "terrain_type": int(terrain_types[i].item()),
            "failed_before_onset": bool(failed_before_onset[i].item()),
            "survived_post_failure_horizon": bool(survived),
            "post_failure_survival_seconds": float(
                own.post_seconds if survived else max(0, done_step - int(onset_step_by_env[i].item())) * dt
            ),
            "recovered_stable_window": recovered,
            "recovery_seconds": recovery_seconds,
            "post_command_vx_rmse": math.sqrt(float(vx_error_sq[i].item() / n)),
            "post_yaw_rate_rmse": math.sqrt(float(yaw_error_sq[i].item() / n)),
            "post_vertical_velocity_rms": math.sqrt(float(vertical_sq[i].item() / n)),
            "post_forward_progress_m": float(forward_progress[i].item()),
            "post_mean_reward_per_step": float(reward_sum[i].item() / n),
            "post_action_near_bound_rate": float(action_near_bound[i].item() / n),
            "post_four_feet_contact_rate": float(four_contact[i].item() / n),
            "post_contact_gated_foot_slip_mps": float(slip[i].item() / n),
        })

    output = Path(own.output).resolve()
    metadata = {
        "schema_version": 1,
        "evaluation": "teacher243_random_onset_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "task": args.task,
        "seed": int(args.seed),
        "num_envs": num_envs,
        "dt": dt,
        "command_x": own.command_x,
        "rates": list(rates),
        "onset_seconds": list(onsets),
        "post_seconds": own.post_seconds,
        "replicates_per_condition": own.replicates_per_condition,
        "policy_mode": own.policy_mode,
        "recovery_definition": {
            "window_seconds": own.recovery_window_seconds,
            "max_abs_vx_error_mps": own.recovery_vx_error,
            "max_abs_yaw_rate_radps": own.recovery_yaw_rate,
        },
    }
    atomic_jsonl(output, metadata, rows)
    print(json.dumps({**metadata, "output": str(output), "robot_rows": len(rows)}, sort_keys=True))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()

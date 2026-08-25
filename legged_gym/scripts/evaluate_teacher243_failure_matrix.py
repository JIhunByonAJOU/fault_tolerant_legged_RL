"""Deterministic per-robot evaluation for fixed actuator degradation cells."""

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


DEFAULT_RATES = (0.2, 0.4, 0.6, 0.8)
NUM_JOINTS = 12


def parse_args():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--checkpoint-path", required=True)
    own.add_argument("--steps", type=int, default=1000)
    own.add_argument("--replicates-per-condition", type=int, default=16)
    own.add_argument("--command-x", type=float, default=0.5)
    own.add_argument("--policy-mode", choices=("teacher", "student"), default="teacher")
    own.add_argument("--rates", type=float, nargs="+", default=list(DEFAULT_RATES))
    own.add_argument(
        "--latent-mode",
        choices=(
            "actual",
            "zero",
            "shuffled",
            "failure_masked",
            "failure_shuffled",
        ),
        default="actual",
    )
    own.add_argument("--output", required=True)
    known, remaining = own.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def condition_for_env(env_id, rates):
    num_conditions = NUM_JOINTS * len(rates)
    condition = int(env_id) % num_conditions
    return condition // len(rates), rates[condition % len(rates)]


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
        "a1_official_wim_teacher243_failure",
        "a1_official_wim_teacher243_failure_fullrange",
        "a1_official_wim_jt_failure_fullrange_onset",
    ) or not args.headless:
        raise ValueError("matrix evaluation requires failure task with --headless")
    if own.policy_mode == "student" and own.latent_mode not in (
        "actual",
        "zero",
        "shuffled",
    ):
        raise ValueError("student policy supports actual/zero/shuffled history modes")
    if own.steps <= 0 or own.replicates_per_condition <= 0:
        raise ValueError("steps and replicates must be positive")
    rates = tuple(float(rate) for rate in own.rates)
    if not rates or any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError("--rates must contain values in [0, 1]")
    if len(set(rates)) != len(rates):
        raise ValueError("--rates must not contain duplicates")
    checkpoint = Path(own.checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    num_conditions = NUM_JOINTS * len(rates)
    num_envs = num_conditions * own.replicates_per_condition
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = num_envs
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

    env_ids = torch.arange(num_envs, device=env.device)
    joint_by_env = torch.tensor(
        [condition_for_env(i, rates)[0] for i in range(num_envs)],
        dtype=torch.long,
        device=env.device,
    )
    rate_by_env = torch.tensor(
        [condition_for_env(i, rates)[1] for i in range(num_envs)],
        dtype=torch.float,
        device=env.device,
    )
    condition_groups = []
    for joint in range(NUM_JOINTS):
        for rate in rates:
            ids = env_ids[(joint_by_env == joint) & torch.isclose(rate_by_env, torch.tensor(rate, device=env.device))]
            condition_groups.append((ids, joint, rate))
    for ids, joint, rate in condition_groups:
        env.set_actuator_degradation(ids, joint, rate)
    initial_effective_strength = env.motor_strength_gt[
        env_ids, joint_by_env
    ].clone()

    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    count = torch.zeros(num_envs, device=env.device)
    reward_sum = torch.zeros(num_envs, device=env.device)
    vx_error_sq = torch.zeros(num_envs, device=env.device)
    yaw_error_sq = torch.zeros(num_envs, device=env.device)
    vertical_sq = torch.zeros(num_envs, device=env.device)
    forward_progress = torch.zeros(num_envs, device=env.device)
    action_abs = torch.zeros(num_envs, device=env.device)
    action_near_bound = torch.zeros(num_envs, device=env.device)
    four_contact = torch.zeros(num_envs, device=env.device)
    airborne = torch.zeros(num_envs, device=env.device)
    slip = torch.zeros(num_envs, device=env.device)
    foot_contacts = torch.zeros(num_envs, 4, device=env.device)
    latent_sum = torch.zeros(num_envs, 8, device=env.device)
    first_done_step = torch.full((num_envs,), own.steps, dtype=torch.long, device=env.device)

    obs = privileged = None
    if own.policy_mode == "student":
        env.commands[:, :] = 0.0
        env.commands[:, 0] = own.command_x
        env.compute_observations()
        obs = env.get_observations()
        privileged = env.get_privileged_observations()

    with torch.inference_mode():
        for step in range(own.steps):
            for ids, joint, rate in condition_groups:
                env.set_actuator_degradation(ids, joint, rate)
            env.commands[:, :] = 0.0
            env.commands[:, 0] = own.command_x
            if own.policy_mode == "teacher":
                env.compute_observations()
                obs = env.get_observations()
                privileged = env.get_privileged_observations()
            if own.policy_mode == "student":
                latent = model.encode_history(obs)
                used_obs = obs
                if own.latent_mode != "actual":
                    used_obs = obs.clone()
                    history = used_obs[:, 235:]
                    if own.latent_mode == "zero":
                        history.zero_()
                    else:
                        used_obs[:, 235:] = torch.roll(history, shifts=1, dims=0)
                actions = model.act_inference_student(used_obs)
            else:
                latent = model.encode_privileged(privileged)
                if own.latent_mode == "actual":
                    used_latent = latent
                elif own.latent_mode == "zero":
                    used_latent = torch.zeros_like(latent)
                elif own.latent_mode == "shuffled":
                    used_latent = torch.roll(latent, shifts=1, dims=0)
                else:
                    counterfactual = privileged.clone()
                    if own.latent_mode == "failure_masked":
                        # Preserve all other privileged context while telling the
                        # encoder that every actuator has its pre-degradation
                        # strength and that no failure is active.
                        counterfactual[:, 2:14] = env.nominal_motor_strength_gt
                        counterfactual[:, 44] = 0.0
                    else:
                        # Preserve physics and all non-failure context, but assign
                        # each robot another robot's joint/severity information.
                        counterfactual[:, 2:14] = torch.roll(
                            privileged[:, 2:14], shifts=1, dims=0
                        )
                        counterfactual[:, 44] = torch.roll(
                            privileged[:, 44], shifts=1, dims=0
                        )
                    used_latent = model.encode_privileged(counterfactual)
                actions = model.act_inference_with_latent(obs, used_latent)

            mask = active.float()
            contacts = env.contact_forces[:, env.feet_indices, 2] > 1.0
            foot_velocity = _refresh_rigid_body_states(
                env, rigid_body_states, torch, "failure matrix step {}".format(step)
            )
            foot_speed = torch.norm(
                foot_velocity[:, :, :2], dim=-1
            )
            count += mask
            vx_error_sq += (env.base_lin_vel[:, 0] - own.command_x).square() * mask
            yaw_error_sq += env.base_ang_vel[:, 2].square() * mask
            vertical_sq += env.base_lin_vel[:, 2].square() * mask
            forward_progress += env.base_lin_vel[:, 0] * env.dt * mask
            action_abs += actions.abs().mean(dim=1) * mask
            action_near_bound += (actions.abs() >= 0.98).float().mean(dim=1) * mask
            contact_count = contacts.sum(dim=1)
            four_contact += (contact_count == 4).float() * mask
            airborne += (contact_count == 0).float() * mask
            foot_contacts += contacts.float() * mask.unsqueeze(1)
            slip += (
                (foot_speed * contacts.float()).sum(dim=1)
                / contacts.float().sum(dim=1).clamp(min=1.0)
            ) * mask
            latent_sum += latent * mask.unsqueeze(1)

            next_obs, next_privileged, rewards, dones, _ = env.step(actions)
            if own.policy_mode == "student":
                obs, privileged = next_obs, next_privileged
            reward_sum += rewards * mask
            newly_done = active & (dones > 0)
            first_done_step[newly_done] = step + 1
            active[newly_done] = False
            if not active.any():
                break

    count = count.clamp(min=1.0)
    terrain_levels = getattr(env, "terrain_levels", torch.full((num_envs,), -1, device=env.device))
    terrain_types = getattr(env, "terrain_types", torch.full((num_envs,), -1, device=env.device))
    rows = []
    for i in range(num_envs):
        n = float(count[i].item())
        joint = int(joint_by_env[i].item())
        rate = float(rate_by_env[i].item())
        rows.append(
            {
                "seed": int(args.seed),
                "env_id": i,
                "replicate": i // num_conditions,
                "joint_index": joint,
                "joint_name": env.dof_names[joint],
                "degradation_rate": rate,
                "remaining_output_fraction": 1.0 - rate,
                "terrain_level": int(terrain_levels[i].item()),
                "terrain_type": int(terrain_types[i].item()),
                "survived_full_horizon": bool(first_done_step[i].item() >= own.steps),
                "survival_steps": int(first_done_step[i].item()),
                "survival_seconds": float(first_done_step[i].item() * env.dt),
                "mean_reward_per_step": float(reward_sum[i].item() / n),
                "command_vx_rmse": math.sqrt(float(vx_error_sq[i].item() / n)),
                "yaw_rate_rmse": math.sqrt(float(yaw_error_sq[i].item() / n)),
                "vertical_velocity_rms": math.sqrt(float(vertical_sq[i].item() / n)),
                "integrated_forward_progress_m": float(forward_progress[i].item()),
                "mean_abs_action": float(action_abs[i].item() / n),
                "action_near_bound_rate": float(action_near_bound[i].item() / n),
                "four_feet_contact_rate": float(four_contact[i].item() / n),
                "airborne_rate": float(airborne[i].item() / n),
                "contact_gated_foot_slip_mps": float(slip[i].item() / n),
                "per_foot_contact_rate": [float(x) for x in (foot_contacts[i] / n).cpu().tolist()],
                "latent_mean": [float(x) for x in (latent_sum[i] / n).cpu().tolist()],
                "effective_motor_strength": float(initial_effective_strength[i].item()),
            }
        )

    output = Path(own.output).resolve()
    metadata = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "task": args.task,
        "seed": int(args.seed),
        "num_envs": num_envs,
        "steps": own.steps,
        "dt": float(env.dt),
        "command_x": own.command_x,
        "rates": list(rates),
        "latent_mode": own.latent_mode,
        "policy_mode": own.policy_mode,
        "joint_indices": list(range(NUM_JOINTS)),
        "joint_names": list(env.dof_names),
        "replicates_per_condition": own.replicates_per_condition,
    }
    atomic_jsonl(output, metadata, rows)
    print(json.dumps({**metadata, "output": str(output), "robot_rows": len(rows)}, sort_keys=True))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()

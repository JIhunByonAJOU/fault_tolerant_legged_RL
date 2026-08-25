"""Read-only deterministic BaseEnv evaluation of one exact checkpoint."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch
import numpy as np
import torch
from isaacgym.torch_utils import get_euler_xyz

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.envs.a1_limping.schema import ActorObservationSlices, schema_manifest
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math import wrap_to_pi


SOURCE_PATHS = (
    "legged_gym/scripts/evaluate_checkpoint_selector.py",
    "legged_gym/scripts/evaluate_checkpoint_sweep.py",
    "legged_gym/evaluation/p1_checkpoint_selector.json",
    "legged_gym/envs/__init__.py",
    "legged_gym/envs/a1_limping/__init__.py",
    "legged_gym/envs/a1_limping/a1_limping.py",
    "legged_gym/envs/a1_limping/a1_limping_config.py",
    "legged_gym/envs/a1_limping/schema.py",
    "legged_gym/envs/base/base_task.py",
    "legged_gym/envs/base/legged_robot.py",
    "legged_gym/envs/base/legged_robot_config.py",
    "legged_gym/learning/teacher_actor_critic.py",
    "legged_gym/learning/teacher_ppo.py",
    "legged_gym/learning/teacher_runner.py",
    "legged_gym/utils/helpers.py",
    "legged_gym/utils/task_registry.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    known, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0]] + remaining
        args = get_args()
    finally:
        sys.argv = original
    for name, value in vars(known).items():
        setattr(args, name.replace("-", "_"), value)
    return args


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise ValueError("{} mismatch: expected {!r}, got {!r}".format(label, expected, actual))


def assert_resolved_contract(resolved, protocol, env_cfg=None, train_cfg=None):
    expected = protocol["resolved_contract"]
    environment = resolved["environment"]
    schema = resolved["schema"]
    recorded = {
        "clip_observations": environment["normalization"]["clip_observations"],
        "clip_actions": environment["normalization"]["clip_actions"],
        "observation_scales": environment["normalization"]["obs_scales"],
        "action_scale": environment["control"]["action_scale"],
        "decimation": environment["control"]["decimation"],
        "stiffness": environment["control"]["stiffness"]["joint"],
        "damping": environment["control"]["damping"]["joint"],
        "actor_observation_dim": schema["actor_observation_dim"],
        "privileged_observation_dim": schema["privileged_observation_dim"],
        "actor_input_dim": schema["actor_input_dim"],
        "critic_input_dim": schema["critic_input_dim"],
        "teacher_latent_dim": schema["teacher_latent_dim"],
    }
    for key, value in recorded.items():
        _assert_equal(value, expected[key], "resolved_config.{}".format(key))
    _assert_equal(resolved["task"], protocol["task"], "resolved task")
    action_coordinate = resolved["attribution"]["action_coordinate"]
    if "environment-only clip" not in action_coordinate:
        raise ValueError("resolved action coordinate is not environment-only clip")
    if env_cfg is not None:
        active = {
            "clip_observations": env_cfg.normalization.clip_observations,
            "clip_actions": env_cfg.normalization.clip_actions,
            "observation_scales": {name: getattr(env_cfg.normalization.obs_scales, name) for name in expected["observation_scales"]},
            "action_scale": env_cfg.control.action_scale,
            "decimation": env_cfg.control.decimation,
            "stiffness": env_cfg.control.stiffness["joint"],
            "damping": env_cfg.control.damping["joint"],
            "actor_observation_dim": env_cfg.env.num_observations,
            "privileged_observation_dim": env_cfg.env.num_privileged_obs,
        }
        for key, value in active.items():
            _assert_equal(value, expected[key], "active config {}".format(key))
    if train_cfg is not None:
        manifest = schema_manifest(env_cfg.env.num_observations)
        _assert_equal(manifest["actor_input_dim"], expected["actor_input_dim"], "active actor input")
        _assert_equal(manifest["critic_input_dim"], expected["critic_input_dim"], "active critic input")
        _assert_equal(train_cfg.policy.teacher_latent_dim, expected["teacher_latent_dim"], "active teacher latent")
    return True


def configure_nominal_baseenv(env_cfg, protocol):
    nominal = protocol["nominal_environment"]
    env_cfg.env.num_envs = protocol["num_envs"]
    env_cfg.env.episode_length_s = protocol["duration_seconds"]
    env_cfg.terrain.mesh_type = nominal["terrain_mesh_type"]
    env_cfg.terrain.curriculum = nominal["terrain_curriculum"]
    env_cfg.commands.curriculum = nominal["command_curriculum"]
    env_cfg.commands.heading_command = nominal["heading_command"]
    env_cfg.noise.add_noise = nominal["observation_noise"]
    env_cfg.domain_rand.push_robots = nominal["pushes"]
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [nominal["friction"], nominal["friction"]]
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.added_mass_range = [nominal["added_mass"], nominal["added_mass"]]
    env_cfg.domain_rand.motor_strength_range = [nominal["motor_strength"], nominal["motor_strength"]]
    env_cfg.domain_rand.kp_scale_range = [nominal["kp_scale"], nominal["kp_scale"]]
    env_cfg.domain_rand.kd_scale_range = [nominal["kd_scale"], nominal["kd_scale"]]
    return env_cfg


def apply_command_bank(env, obs, command_bank):
    env.commands[:, :3] = command_bank
    obs[:, ActorObservationSlices.COMMAND] = command_bank * env.commands_scale


def environment_clip(raw_actions, clip_actions=1.0):
    return torch.clamp(raw_actions, -float(clip_actions), float(clip_actions))


def quantile_summary(values):
    values = torch.as_tensor(values).detach().cpu().numpy().astype(np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("quantile values must be finite and non-empty")
    return {"median": float(np.quantile(values, 0.5)), "p10": float(np.quantile(values, 0.1)), "p90": float(np.quantile(values, 0.9))}


def summarize_raw_and_applied_actions(raw_abs_sum, raw_abs_max, raw_gt_clip, applied_abs_sum, applied_near, action_count):
    denom = action_count.clamp_min(1.0)
    return {
        "raw_abs_mean": quantile_summary(raw_abs_sum / denom),
        "raw_abs_max": float(raw_abs_max.max().item()),
        "fraction_abs_raw_gt_1": quantile_summary(raw_gt_clip / denom),
        "applied_abs_mean": quantile_summary(applied_abs_sum / denom),
        "applied_action_near_bound_rate": quantile_summary(applied_near / denom),
    }


def _repository_provenance():
    root = Path(LEGGED_GYM_ROOT_DIR).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root, text=True, capture_output=True, check=True).stdout
    diff = subprocess.run(["git", "diff", "--binary", "HEAD", "--"], cwd=root, capture_output=True, check=True).stdout
    sources = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError("required source missing: {}".format(relative))
        sources[relative] = sha256_file(path)
    return {"repository_head": head, "repository_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(), "repository_diff_sha256": hashlib.sha256(diff).hexdigest(), "sources": sources}


def _group_summary(indices, tensors, survival, correct_yaw_sign):
    metrics = {name: quantile_summary(value[indices]) for name, value in tensors.items()}
    return {"num_envs": int(indices.sum().item()), "finite": all(math.isfinite(value) for summary in metrics.values() for value in summary.values()), "survival_rate": float(survival[indices].float().mean().item()), "falls": int((~survival[indices]).sum().item()), "correct_yaw_sign_rate": float(correct_yaw_sign[indices].float().mean().item()), "metrics": metrics}


def evaluate(args):
    if not args.headless or args.task != "a1_limping_base_v2":
        raise ValueError("selector requires --headless --task a1_limping_base_v2")
    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.is_file() or not __import__("re").fullmatch(r"model_\d+\.pt", checkpoint_path.name):
        raise ValueError("an exact model_<iteration>.pt --checkpoint-path is required")
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise ValueError("output already exists: {}".format(output_path))
    protocol_path = Path(args.protocol).resolve()
    resolved_path = Path(args.resolved_config).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if protocol.get("name") != "p1_baseenv_checkpoint_selector_v1":
        raise ValueError("unexpected selector protocol")
    assert_resolved_contract(resolved, protocol)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_nominal_baseenv(env_cfg, protocol)
    assert_resolved_contract(resolved, protocol, env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if env.graphics_device_id != -1 or env.viewer is not None:
        raise RuntimeError("headless selector created graphics or viewer")
    try:
        train_cfg.runner.resume = False
        runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
        runner.load(str(checkpoint_path), load_optimizer=False)
        actor_critic = runner.alg.actor_critic
        actor_critic.eval()
        obs, privileged_obs = env.reset()
        num_envs = env.num_envs
        device = env.device
        command_values = torch.tensor([item["command"] for item in protocol["command_groups"]], dtype=env.commands.dtype, device=device)
        group_ids = torch.arange(num_envs, device=device) % len(command_values)
        command_bank = command_values[group_ids]
        protocol_dt = protocol["control_dt_seconds"]
        total_steps = int(round(protocol["duration_seconds"] / protocol_dt))
        warmup_steps = int(round(protocol["warmup_seconds"] / protocol_dt))
        if total_steps != 1000 or warmup_steps != 100 or not math.isclose(env.dt, protocol_dt, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("active control horizon does not match frozen protocol: dt={!r}, total_steps={}, warmup_steps={}".format(env.dt, total_steps, warmup_steps))
        zeros = lambda: torch.zeros(num_envs, dtype=torch.float, device=device)
        fallen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        measured = zeros()
        planar_error_sq, yaw_error_sq, vertical_sq = zeros(), zeros(), zeros()
        four, diagonal, airborne = zeros(), zeros(), zeros()
        raw_abs_sum, raw_abs_max, raw_gt_clip = zeros(), zeros(), zeros()
        applied_abs_sum, applied_near, action_count = zeros(), zeros(), zeros()
        progress, path_length = zeros(), zeros()
        start_xy = previous_xy = final_xy = None
        previous_yaw = None
        yaw_change = zeros()
        with torch.inference_mode():
            for step in range(total_steps):
                apply_command_bank(env, obs, command_bank)
                raw_actions = actor_critic.act_inference_raw(obs, privileged_obs)
                applied_actions = environment_clip(raw_actions, protocol["resolved_contract"]["clip_actions"])
                obs, privileged_obs, _, dones, infos = env.step(raw_actions)
                time_outs = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool)).bool()
                fallen |= dones.bool() & ~time_outs
                if step < warmup_steps:
                    continue
                active = ~fallen
                xy = env.root_states[:, :2]
                _, _, yaw = get_euler_xyz(env.base_quat)
                yaw = wrap_to_pi(yaw)
                if start_xy is None:
                    start_xy, previous_xy, final_xy = xy.clone(), xy.clone(), xy.clone()
                    previous_yaw = yaw.clone()
                measured[active] += 1.0
                planar_error_sq[active] += torch.square(env.base_lin_vel[active, :2] - command_bank[active, :2]).sum(dim=1)
                yaw_error_sq[active] += torch.square(env.base_ang_vel[active, 2] - command_bank[active, 2])
                vertical_sq[active] += torch.square(env.base_lin_vel[active, 2])
                speed = torch.norm(command_bank[:, :2], dim=1).clamp_min(1e-12)
                direction = command_bank[:, :2] / speed.unsqueeze(1)
                progress[active] += (env.base_lin_vel[active, :2] * direction[active]).sum(dim=1) * env.dt
                delta_xy = torch.norm(xy - previous_xy, dim=1)
                path_length[active] += delta_xy[active]
                final_xy[active] = xy[active]
                delta_yaw = wrap_to_pi(yaw - previous_yaw)
                yaw_change[active] += delta_yaw[active]
                previous_xy[active] = xy[active]
                previous_yaw[active] = yaw[active]
                contacts = env.contact_forces[:, env.feet_indices, 2] > env.cfg.privileged.foot_contact_force_threshold
                count = contacts.sum(dim=1)
                four[active] += (count[active] == 4).float()
                airborne[active] += (count[active] == 0).float()
                diag = (count == 2) & ((contacts[:, 0] & contacts[:, 3]) | (contacts[:, 1] & contacts[:, 2]))
                diagonal[active] += diag[active].float()
                raw_abs = raw_actions.abs()
                applied_abs = applied_actions.abs()
                raw_abs_sum[active] += raw_abs[active].sum(dim=1)
                raw_abs_max[active] = torch.maximum(raw_abs_max[active], raw_abs[active].max(dim=1).values)
                raw_gt_clip[active] += (raw_abs[active] > 1.0).float().sum(dim=1)
                applied_abs_sum[active] += applied_abs[active].sum(dim=1)
                applied_near[active] += (applied_abs[active] >= protocol["action_near_bound_threshold"]).float().sum(dim=1)
                action_count[active] += raw_actions.shape[1]
        denom = measured.clamp_min(1.0)
        measured_horizon = protocol["duration_seconds"] - protocol["warmup_seconds"]
        net_displacement = torch.norm(final_xy - start_xy, dim=1)
        tensors = {
            "planar_command_rmse_mps": torch.sqrt(planar_error_sq / denom),
            "yaw_rate_rmse_radps": torch.sqrt(yaw_error_sq / denom),
            "command_direction_progress_ratio": progress / (torch.norm(command_bank[:, :2], dim=1) * measured_horizon),
            "path_efficiency": net_displacement / path_length.clamp_min(1e-6),
            "vertical_velocity_rms_mps": torch.sqrt(vertical_sq / denom),
            "four_feet_contact_rate": four / denom,
            "diagonal_contact_rate": diagonal / denom,
            "airborne_rate": airborne / denom,
            "raw_abs_mean": raw_abs_sum / action_count.clamp_min(1.0),
            "raw_abs_max": raw_abs_max,
            "fraction_abs_raw_gt_1": raw_gt_clip / action_count.clamp_min(1.0),
            "applied_abs": applied_abs_sum / action_count.clamp_min(1.0),
            "applied_abs_mean": applied_abs_sum / action_count.clamp_min(1.0),
            "applied_action_near_bound_rate": applied_near / action_count.clamp_min(1.0),
        }
        survival = ~fallen
        target_sign = torch.sign(command_bank[:, 2])
        correct_yaw_sign = (target_sign == 0) | (torch.sign(yaw_change) == target_sign)
        per_command = {}
        for index, item in enumerate(protocol["command_groups"]):
            per_command[item["name"]] = {"command": item["command"], **_group_summary(group_ids == index, tensors, survival, correct_yaw_sign)}
        aggregate = _group_summary(torch.ones(num_envs, dtype=torch.bool, device=device), tensors, survival, correct_yaw_sign)
        aggregate["action_diagnostics"] = summarize_raw_and_applied_actions(raw_abs_sum, raw_abs_max, raw_gt_clip, applied_abs_sum, applied_near, action_count)
        source_provenance = _repository_provenance()
        iteration = int(checkpoint_path.stem.split("_")[1])
        result = {
            "schema_version": 1,
            "finite": aggregate["finite"] and all(group["finite"] for group in per_command.values()),
            "task": args.task,
            "seed": protocol["seed"],
            "num_envs": num_envs,
            "checkpoint": {"iteration": iteration, "path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path), "name": protocol["name"]},
            "resolved_config": {"path": str(resolved_path), "sha256": sha256_file(resolved_path)},
            "normalization_action_contract": {"observation_clip": 100.0, "observation_scales": protocol["resolved_contract"]["observation_scales"], "policy_action": "raw_deterministic_actor_mean", "policy_tanh": False, "application_clip": "BaseEnv_only", "action_clip": 1.0},
            "source_provenance": source_provenance,
            "per_command": per_command,
            "aggregate": aggregate,
        }
        if not result["finite"]:
            raise ValueError("selector produced nonfinite metrics")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp-" + uuid.uuid4().hex)
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(output_path))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return result
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    evaluate(_parse_args())

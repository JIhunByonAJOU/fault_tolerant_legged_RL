"""One-cell fresh-process evaluator for the frozen official WIM protocol."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_and_verify_protocol(path):
    from legged_gym.scripts.evaluate_official_wim_sweep import load_protocol

    return load_protocol(path)


def apply_episode_constant_commands(env):
    import torch
    env.commands[:, 0] = 0.75
    env.commands[:, 1] = torch.linspace(-0.1, 0.1, env.num_envs, device=env.device)
    env.commands[:, 2] = 0.0
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0


def _refresh_rigid_body_states(env, rigid_body_states, torch, context):
    try:
        env.gym.refresh_rigid_body_state_tensor(env.sim)
    except Exception as exc:
        raise RuntimeError("{}: failed to refresh rigid-body state tensor".format(context)) from exc
    expected_shape = (env.num_envs, env.num_bodies, 13)
    if tuple(rigid_body_states.shape) != expected_shape:
        raise RuntimeError(
            "{}: rigid-body state shape {} != {}".format(
                context, tuple(rigid_body_states.shape), expected_shape
            )
        )
    foot_velocity = rigid_body_states[:, env.feet_indices, 7:10]
    expected_foot_shape = (env.num_envs, 4, 3)
    if tuple(foot_velocity.shape) != expected_foot_shape:
        raise RuntimeError(
            "{}: selected foot velocity shape {} != {}".format(
                context, tuple(foot_velocity.shape), expected_foot_shape
            )
        )
    if not bool(torch.isfinite(foot_velocity).all()):
        raise RuntimeError("{}: selected foot velocity contains nonfinite values".format(context))
    return foot_velocity


def _acquire_rigid_body_states(env, gymtorch, torch):
    try:
        descriptor = env.gym.acquire_rigid_body_state_tensor(env.sim)
        wrapped = gymtorch.wrap_tensor(descriptor)
    except Exception as exc:
        raise RuntimeError("failed to acquire or wrap rigid-body state tensor") from exc
    expected_numel = env.num_envs * env.num_bodies * 13
    if wrapped.numel() != expected_numel:
        raise RuntimeError(
            "rigid-body state element count {} != {}".format(wrapped.numel(), expected_numel)
        )
    try:
        rigid_body_states = wrapped.view(env.num_envs, env.num_bodies, 13)
    except Exception as exc:
        raise RuntimeError("failed to view rigid-body state tensor with the expected layout") from exc
    if tuple(rigid_body_states.shape) != (env.num_envs, env.num_bodies, 13):
        raise RuntimeError("rigid-body state tensor has an unexpected shape")
    feet_indices = env.feet_indices
    if feet_indices.numel() != 4:
        raise RuntimeError("expected exactly four foot body indices")
    if bool((feet_indices < 0).any()) or bool((feet_indices >= env.num_bodies).any()):
        raise RuntimeError("foot body index is outside the rigid-body state range")
    _refresh_rigid_body_states(env, rigid_body_states, torch, "initial rigid-body state")
    if not bool(torch.isfinite(rigid_body_states).all()):
        raise RuntimeError("initial rigid-body state tensor contains nonfinite values")
    return rigid_body_states


class _TerrainArguments(dict):
    @property
    def terrain_kwargs(self):
        return self


def _configure_frozen_terrain(env_cfg, terrain):
    from legged_gym.utils import terrain as terrain_module
    from isaacgym import terrain_utils

    arguments = dict(terrain["arguments"])
    generator = terrain["generator"]
    if generator == "pyramid_sloped_plus_random_uniform":
        def official_wim_rough_slope(surface, **kwargs):
            terrain_utils.pyramid_sloped_terrain(surface, slope=kwargs["slope"], platform_size=kwargs["platform_size"])
            terrain_utils.random_uniform_terrain(surface, min_height=kwargs["min_height"], max_height=kwargs["max_height"], step=kwargs["step"], downsampled_scale=kwargs["downsampled_scale"])
        terrain_module.official_wim_rough_slope = official_wim_rough_slope
        generator = "official_wim_rough_slope"
    elif not generator.startswith("terrain_utils."):
        generator = "terrain_utils." + generator
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    env_cfg.terrain.terrain_kwargs = _TerrainArguments(type=generator, **arguments)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1


def evaluate_checkpoint_cell(args):
    import torch
    from isaacgym import gymtorch
    from rsl_rl.runners import OnPolicyRunner
    from legged_gym.envs import task_registry
    from legged_gym.evaluation.official_wim_metrics import summarize_cell
    from legged_gym.utils.helpers import class_to_dict

    protocol = load_and_verify_protocol(args.protocol)
    resolved = json.loads(Path(args.resolved_config).read_text(encoding="utf-8"))
    protocol_sha = _sha(args.protocol)
    resolved_sha = _sha(args.resolved_config)
    if protocol_sha != args.expected_protocol_sha256:
        raise ValueError("protocol hash mismatch")
    if resolved_sha != args.expected_resolved_config_sha256:
        raise ValueError("resolved configuration hash mismatch")
    if args.task != protocol["task"] or resolved.get("task") != args.task:
        raise ValueError("task/protocol/resolved mismatch")
    checkpoint = Path(args.checkpoint_path).resolve()
    before = _sha(checkpoint)
    if before != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")
    choices = protocol["implementation_choice_agent_recommendation"]
    matching_terrains = [item for item in choices["terrain_cells"] if item["name"] == args.terrain]
    if len(matching_terrains) != 1:
        raise ValueError("unknown or duplicate frozen terrain")
    if args.seed not in choices["seeds"] or isinstance(args.seed, bool):
        raise ValueError("seed is outside the frozen protocol")
    terrain = matching_terrains[0]
    expected_index = choices["terrain_cells"].index(terrain) * len(choices["seeds"]) + choices["seeds"].index(args.seed)
    if args.cell_index != expected_index:
        raise ValueError("cell index does not match frozen terrain/seed order")

    args.num_envs = 64
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    _configure_frozen_terrain(env_cfg, terrain)
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1000.0
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    rigid_body_states = _acquire_rigid_body_states(env, gymtorch, torch)
    runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=args.rl_device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    observations = env.get_observations()
    starts = env.root_states[:, 0].clone()
    base_failed = torch.zeros(64, dtype=torch.bool, device=env.device)
    crossed = torch.zeros_like(base_failed)
    velocity_error_sum = torch.zeros(64, device=env.device)
    lateral_error_sum = torch.zeros(64, device=env.device)
    yaw_drift_sum = torch.zeros(64, device=env.device)
    roll_pitch_sum = torch.zeros(64, 2, device=env.device)
    base_height_sum = torch.zeros(64, device=env.device)
    foot_slip_sum = torch.zeros(64, device=env.device)
    duty_sum = torch.zeros(64, 4, device=env.device)
    contact_count_sum = torch.zeros(64, 5, device=env.device)
    diagonal_pair_sum = torch.zeros(64, device=env.device)
    raw_action_abs_sum = torch.zeros(64, device=env.device)
    applied_action_abs_sum = torch.zeros(64, device=env.device)
    action_near_bound_sum = torch.zeros(64, device=env.device)
    first_cross_step = torch.full((64,), -1, dtype=torch.long, device=env.device)
    steps = int(round(choices["episode_horizon_seconds"] / env.dt))
    from isaacgym.torch_utils import get_euler_xyz
    for step_index in range(steps):
        apply_episode_constant_commands(env)
        with torch.inference_mode():
            raw_actions = policy(observations)
        observations, _, _, _, _ = env.step(raw_actions)
        foot_velocity = _refresh_rigid_body_states(
            env,
            rigid_body_states,
            torch,
            "terrain {} seed {} step {}".format(terrain["name"], args.seed, step_index),
        )
        base_contact = torch.any(torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        base_failed |= base_contact
        new_cross = (env.root_states[:, 0] - starts >= 8.0) & ~base_failed & ~crossed
        first_cross_step[new_cross] = step_index + 1
        crossed |= new_cross
        velocity_error_sum += torch.sum((env.base_lin_vel[:, :2] - env.commands[:, :2]) ** 2, dim=1)
        lateral_error_sum += torch.abs(env.root_states[:, 1] - env.env_origins[:, 1])
        roll, pitch, yaw = get_euler_xyz(env.base_quat)
        yaw_drift_sum += torch.abs(yaw)
        roll_pitch_sum[:, 0] += torch.abs(roll)
        roll_pitch_sum[:, 1] += torch.abs(pitch)
        base_height_sum += env.root_states[:, 2] - env.env_origins[:, 2]
        contacts = env.contact_forces[:, env.feet_indices, 2] > 1.0
        duty_sum += contacts.float()
        count = contacts.sum(dim=1)
        diagonal_pair_sum += (((contacts[:, 0] & contacts[:, 3]) | (contacts[:, 1] & contacts[:, 2])) & (count == 2)).float()
        for contact_number in range(5):
            contact_count_sum[:, contact_number] += (count == contact_number).float()
        foot_slip_sum += (torch.norm(foot_velocity[:, :, :2], dim=2) * contacts.float()).mean(dim=1)
        raw_action_abs_sum += raw_actions.abs().mean(dim=1)
        applied_action_abs_sum += env.actions.abs().mean(dim=1)
        action_near_bound_sum += (env.actions.abs() >= 99.0).float().mean(dim=1)
    progress = (env.root_states[:, 0] - starts) / 8.0
    episodes = [{
        "success": bool(crossed[i]), "base_contact": bool(base_failed[i]), "timeout": not bool(crossed[i]) and not bool(base_failed[i]),
        "along_track_progress_m": float(progress[i] * 8.0), "progress_ratio": float(progress[i]),
        "time_to_cross_s": float(first_cross_step[i]) * env.dt if first_cross_step[i] >= 0 else choices["episode_horizon_seconds"],
        "command_velocity_rmse": math.sqrt(float(velocity_error_sum[i]) / steps),
        "mean_lateral_error_m": float(lateral_error_sum[i] / steps), "mean_yaw_drift_rad": float(yaw_drift_sum[i] / steps),
        "mean_abs_roll_rad": float(roll_pitch_sum[i, 0] / steps), "mean_abs_pitch_rad": float(roll_pitch_sum[i, 1] / steps),
        "mean_base_height_m": float(base_height_sum[i] / steps), "mean_foot_slip_mps": float(foot_slip_sum[i] / steps),
        "duty_factors": [float(value / steps) for value in duty_sum[i]],
        "airborne_fraction": float(contact_count_sum[i, 0] / steps), "two_contact_fraction": float(contact_count_sum[i, 2] / steps),
        "diagonal_pair_fraction": float(diagonal_pair_sum[i] / steps), "four_contact_fraction": float(contact_count_sum[i, 4] / steps),
        "mean_raw_action_abs": float(raw_action_abs_sum[i] / steps), "mean_applied_action_abs": float(applied_action_abs_sum[i] / steps),
        "applied_action_near_bound_rate": float(action_near_bound_sum[i] / steps),
    } for i in range(64)]
    summary = summarize_cell(episodes)
    for key in ("along_track_progress_m", "time_to_cross_s", "mean_lateral_error_m", "mean_yaw_drift_rad", "mean_abs_roll_rad", "mean_abs_pitch_rad", "mean_base_height_m", "mean_foot_slip_mps", "airborne_fraction", "two_contact_fraction", "diagonal_pair_fraction", "four_contact_fraction", "mean_raw_action_abs", "mean_applied_action_abs", "applied_action_near_bound_rate"):
        summary[key] = sum(item[key] for item in episodes) / len(episodes)
    summary["duty_factors"] = [sum(item["duty_factors"][foot] for item in episodes) / len(episodes) for foot in range(4)]
    summary["timeout_rate"] = sum(item["timeout"] for item in episodes) / len(episodes)
    summary.update({
        "checkpoint": {"iteration": int(checkpoint.stem.split("_")[1]), "name": checkpoint.name, "path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": before},
        "protocol_sha256": protocol_sha,
        "resolved_config_sha256": resolved_sha,
        "terrain": terrain["name"],
        "terrain_config": terrain,
        "seed": args.seed,
        "cell_index": args.cell_index,
        "child_provenance": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "evaluator_sha256": _sha(__file__),
        },
    })
    if _sha(checkpoint) != before:
        raise RuntimeError("checkpoint changed during evaluation")
    return summary


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--checkpoint-path", required=True)
    own.add_argument("--resolved-config", required=True)
    own.add_argument("--protocol", required=True)
    own.add_argument("--terrain", required=True)
    own.add_argument("--seed", required=True, type=int)
    own.add_argument("--cell-index", required=True, type=int)
    own.add_argument("--expected-checkpoint-sha256", required=True)
    own.add_argument("--expected-protocol-sha256", required=True)
    own.add_argument("--expected-resolved-config-sha256", required=True)
    own.add_argument("--output", required=True)
    single_value_flags = (
        "--checkpoint-path", "--resolved-config", "--protocol", "--terrain",
        "--seed", "--cell-index", "--expected-checkpoint-sha256",
        "--expected-protocol-sha256", "--expected-resolved-config-sha256", "--output",
    )
    if any(raw_argv.count(flag) != 1 for flag in single_value_flags):
        raise ValueError("single-cell request requires every cell argument exactly once")
    known, remaining = own.parse_known_args(raw_argv)
    saved = sys.argv
    try:
        sys.argv = [saved[0]] + remaining
        import isaacgym  # must precede helpers, which imports torch
        import legged_gym.envs
        from legged_gym.utils.helpers import get_args
        args = get_args()
    finally:
        sys.argv = saved
    for key, value in vars(known).items():
        setattr(args, key.replace("-", "_"), value)
    cell = evaluate_checkpoint_cell(args)
    _atomic(known.output, cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

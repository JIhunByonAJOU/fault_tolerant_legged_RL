"""Finite, headless, single-scenario teacher gait evaluation."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch
import torch
from isaacgym.torch_utils import get_euler_xyz

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.envs.a1_limping.schema import ActorObservationSlices
from legged_gym.evaluation.gait_metrics import (
    ARC_SCENARIOS,
    HELDOUT_SCENARIO,
    SCENARIO_COMMANDS,
    ZERO_YAW_MOVING,
    ContactCycleAccumulator,
    atomic_write_json,
    gate_cell,
    heldout_command_sequence,
    load_gate,
    quantile_summary,
    sha256_file,
)
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym.utils.math import wrap_to_pi


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_COMMANDS), required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True)
    known, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0]] + remaining
        args = get_args()
    finally:
        sys.argv = original
    args.scenario = known.scenario
    args.gate = known.gate
    args.output = known.output
    return args


def _source_preflight_record(load_run):
    path = Path(str(load_run)) / "source_preflight.json"
    run_preflight = (
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        if path.is_file()
        else None
    )
    sources = {
        "evaluate_teacher.py": Path(__file__).resolve(),
        "gait_metrics.py": Path(__file__).resolve().parents[1]
        / "evaluation"
        / "gait_metrics.py",
        "evaluate_gait_matrix.py": Path(__file__).resolve().with_name(
            "evaluate_gait_matrix.py"
        ),
    }
    return {
        "run_source_preflight": run_preflight,
        "current_sources": {
            name: {"path": str(source), "sha256": sha256_file(source)}
            for name, source in sources.items()
        },
    }


def _summary_map(**values):
    return {name: quantile_summary(value.detach().cpu()) for name, value in values.items()}


def evaluate(args):
    if not args.headless:
        raise ValueError("evaluate_teacher requires --headless; use view_teacher.py for viewing")
    if args.task != "a1_limping_base_v2":
        raise ValueError("the P1 gait evaluator supports a1_limping_base_v2 only")
    if args.scenario not in SCENARIO_COMMANDS:
        raise ValueError("exactly one known --scenario is required")

    gate_path = Path(args.gate).resolve()
    gate = load_gate(gate_path)
    if args.num_envs != gate["num_envs"]:
        raise ValueError("--num_envs must equal gate num_envs={}".format(gate["num_envs"]))
    if args.seed not in gate["seeds"]:
        raise ValueError("--seed must be one of {}".format(gate["seeds"]))

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = gate["num_envs"]
    env_cfg.env.episode_length_s = gate["protocol"]["duration_seconds"]
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = list(gate["protocol"]["friction_range"])
    command = SCENARIO_COMMANDS[args.scenario]
    env_cfg.commands.straight_command_probability = 0.0
    env_cfg.commands.ranges.lin_vel_x = [command[0], command[0]]
    env_cfg.commands.ranges.lin_vel_y = [command[1], command[1]]
    env_cfg.commands.ranges.ang_vel_yaw = [command[2], command[2]]
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if env.graphics_device_id != -1 or env.viewer is not None:
        raise RuntimeError("headless evaluator created graphics or a viewer")

    train_cfg.runner.resume = True
    load_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    checkpoint_path = get_load_path(
        load_root, load_run=args.load_run, checkpoint=args.checkpoint
    )
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    actor_critic = runner.alg.actor_critic
    obs, privileged_obs = env.reset()

    num_envs = env.num_envs
    device = env.device
    total_steps = int(round(gate["protocol"]["duration_seconds"] / env.dt))
    warmup_steps = int(round(gate["protocol"]["warmup_seconds"] / env.dt))
    measured_horizon = gate["protocol"]["duration_seconds"] - gate["protocol"]["warmup_seconds"]
    zeros = lambda: torch.zeros(num_envs, dtype=torch.float, device=device)
    ever_fallen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    measured_steps = zeros()
    planar_speed_sum = zeros()
    abs_yaw_rate_sum = zeros()
    planar_error_sq = zeros()
    yaw_error_sq = zeros()
    vertical_velocity_sq = zeros()
    foot_slip_sum = zeros()
    diagonal_contact_sum = zeros()
    four_contact_sum = zeros()
    airborne_sum = zeros()
    action_near_count = zeros()
    action_count = zeros()
    path_length = zeros()
    yaw_change = zeros()
    measurement_start_xy = None
    previous_xy = None
    final_xy = None
    measurement_start_yaw = None
    previous_yaw = None
    min_height = None
    max_height = None
    swing_duration = gate["thresholds"]["gait_cycle"]["valid_swing_duration_seconds"]
    duty_range = gate["thresholds"]["gait_cycle"]["env_foot_duty_range"]
    contact_filter = gate["protocol"]["contact_filter"]
    cycles = ContactCycleAccumulator(
        num_envs,
        4,
        env.dt,
        device=device,
        valid_swing_duration=(swing_duration["min"], swing_duration["max"]),
        contact_force_threshold=contact_filter["force_threshold_n"],
        env_foot_duty_range=(duty_range["min"], duty_range["max"]),
        environment_valid_swing_count_min=gate["thresholds"]["gait_cycle"][
            "valid_swing_count_p10_min"
        ],
    )
    fixed_command = torch.tensor(command, dtype=env.commands.dtype, device=device).view(1, 3)
    heldout_commands = None
    heldout_segment_steps = None
    target_xy = None
    target_yaw = None
    target_path_length = zeros()
    target_error_sq = zeros()
    if args.scenario == HELDOUT_SCENARIO:
        generator = gate["protocol"]["heldout_command_generator"]
        heldout_commands = torch.as_tensor(
            heldout_command_sequence(args.seed, num_envs, generator),
            dtype=env.commands.dtype,
            device=device,
        )
        heldout_segment_steps = int(round(generator["segment_seconds"] / env.dt))

    with torch.inference_mode():
        for step in range(total_steps):
            if heldout_commands is None:
                step_commands = fixed_command.expand(num_envs, -1)
            else:
                segment = min(step // heldout_segment_steps, heldout_commands.shape[0] - 1)
                step_commands = heldout_commands[segment]
                env.commands[:, :3] = step_commands
                obs[:, ActorObservationSlices.COMMAND] = step_commands * env.commands_scale
            raw_actions = actor_critic.act_inference_raw(obs, privileged_obs)
            actions = torch.tanh(raw_actions)
            obs, privileged_obs, _, dones, infos = env.step(actions)
            time_outs = infos.get(
                "time_outs", torch.zeros_like(dones, dtype=torch.bool)
            ).bool()
            ever_fallen |= dones.bool() & ~time_outs
            if step < warmup_steps:
                continue
            active = ~ever_fallen
            xy = env.root_states[:, :2]
            _, _, yaw = get_euler_xyz(env.base_quat)
            yaw = wrap_to_pi(yaw)
            if measurement_start_xy is None:
                measurement_start_xy = xy.clone()
                previous_xy = xy.clone()
                final_xy = xy.clone()
                measurement_start_yaw = yaw.clone()
                previous_yaw = yaw.clone()
                min_height = env.root_states[:, 2].clone()
                max_height = min_height.clone()
                target_xy = xy.clone()
                target_yaw = yaw.clone()

            measured_steps[active] += 1.0
            planar_speed_sum[active] += torch.norm(env.base_lin_vel[active, :2], dim=1)
            abs_yaw_rate_sum[active] += torch.abs(env.base_ang_vel[active, 2])
            planar_error_sq[active] += torch.sum(
                torch.square(env.base_lin_vel[active, :2] - step_commands[active, :2]), dim=1
            )
            yaw_error_sq[active] += torch.square(
                env.base_ang_vel[active, 2] - step_commands[active, 2]
            )
            vertical_velocity_sq[active] += torch.square(env.base_lin_vel[active, 2])
            delta_xy = torch.norm(xy - previous_xy, dim=1)
            path_length[active] += delta_xy[active]
            final_xy[active] = xy[active]
            delta_yaw = wrap_to_pi(yaw - previous_yaw)
            yaw_change[active] += delta_yaw[active]
            previous_xy[active] = xy[active]
            previous_yaw[active] = yaw[active]
            height = env.root_states[:, 2]
            min_height[active] = torch.minimum(min_height[active], height[active])
            max_height[active] = torch.maximum(max_height[active], height[active])
            contacts = env.contact_forces[:, env.feet_indices, 2]
            contact_mask = contacts > contact_filter["force_threshold_n"]
            contact_count = contact_mask.sum(dim=1)
            diagonal = (contact_count == 2) & (
                (contact_mask[:, 0] & contact_mask[:, 3])
                | (contact_mask[:, 1] & contact_mask[:, 2])
            )
            diagonal_contact_sum[active] += diagonal[active].float()
            four_contact_sum[active] += (contact_count[active] == 4).float()
            airborne_sum[active] += (contact_count[active] == 0).float()
            cycles.update(contacts, active=active)
            action_near_count[active] += (
                torch.abs(actions[active])
                >= gate["protocol"]["action_near_bound_threshold"]
            ).float().sum(dim=1)
            action_count[active] += actions.shape[1]
            diagnostics = env.get_rollout_diagnostics()
            foot_slip_sum[active] += diagnostics["foot_slip"][active]
            if heldout_commands is not None:
                cos_yaw = torch.cos(target_yaw)
                sin_yaw = torch.sin(target_yaw)
                world_vx = cos_yaw * step_commands[:, 0] - sin_yaw * step_commands[:, 1]
                world_vy = sin_yaw * step_commands[:, 0] + cos_yaw * step_commands[:, 1]
                target_xy[:, 0] += world_vx * env.dt
                target_xy[:, 1] += world_vy * env.dt
                target_yaw += step_commands[:, 2] * env.dt
                target_path_length[active] += (
                    torch.norm(step_commands[active, :2], dim=1) * env.dt
                )
                target_error_sq[active] += torch.sum(
                    torch.square(xy[active] - target_xy[active]), dim=1
                )

    denom = measured_steps.clamp_min(1.0)
    displacement = final_xy - measurement_start_xy
    net_displacement = torch.norm(displacement, dim=1)
    command_xy = torch.tensor(command[:2], dtype=torch.float, device=device)
    command_speed = torch.norm(command_xy)
    if command_speed > 0:
        direction = command_xy / command_speed
        progress = torch.sum(displacement * direction.view(1, 2), dim=1)
        progress_ratio = progress / (command_speed * measured_horizon)
    else:
        progress_ratio = torch.zeros_like(net_displacement)
    integrated_target_yaw = command[2] * measured_horizon
    normalization_floor = gate["protocol"]["heldout_command_generator"][
        "trajectory_rmse_normalization_floor_m"
    ]
    normalized_trajectory_rmse = torch.sqrt(target_error_sq / denom) / (
        target_path_length.clamp_min(normalization_floor)
    )
    metrics = _summary_map(
        planar_speed_mps=planar_speed_sum / denom,
        abs_yaw_rate_radps=abs_yaw_rate_sum / denom,
        drift_m=net_displacement,
        planar_command_rmse_mps=torch.sqrt(planar_error_sq / denom),
        yaw_rate_rmse_radps=torch.sqrt(yaw_error_sq / denom),
        vertical_velocity_rms_mps=torch.sqrt(vertical_velocity_sq / denom),
        base_height_peak_to_peak_m=max_height - min_height,
        foot_slip_cost=foot_slip_sum / denom,
        diagonal_contact_rate=diagonal_contact_sum / denom,
        four_feet_contact_rate=four_contact_sum / denom,
        airborne_rate=airborne_sum / denom,
        action_near_bound_rate=action_near_count / action_count.clamp_min(1.0),
        abs_net_yaw_rad=torch.abs(yaw_change),
        path_efficiency=net_displacement / path_length.clamp_min(1.0e-6),
        command_direction_progress_ratio=progress_ratio,
        abs_integrated_yaw_error_rad=torch.abs(yaw_change - integrated_target_yaw),
        normalized_trajectory_rmse=normalized_trajectory_rmse,
    )
    correct_yaw_sign_rate = 1.0
    if args.scenario in ARC_SCENARIOS:
        correct_yaw_sign_rate = (
            torch.sign(yaw_change) == (1 if command[2] > 0 else -1)
        ).float().mean().item()
    result = {
        "schema_version": 1,
        "scenario": args.scenario,
        "seed": args.seed,
        "num_envs": num_envs,
        "return_code": 0,
        "finite": all(
            torch.isfinite(torch.tensor(list(summary.values()))).all().item()
            for summary in metrics.values()
        ),
        "survival_rate": (~ever_fallen).float().mean().item(),
        "correct_yaw_sign_rate": correct_yaw_sign_rate,
        "metrics": metrics,
        "gait_cycles": cycles.result(),
        "protocol": gate["protocol"],
        "target_command": list(command),
        "heldout_seed_namespace": (
            gate["protocol"]["heldout_command_generator"]["namespace"]
            if args.scenario == HELDOUT_SCENARIO
            else None
        ),
        "gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
        "gate_sha256": sha256_file(gate_path),
        "checkpoint": {
            "path": str(Path(checkpoint_path).resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "source_preflight": _source_preflight_record(args.load_run),
    }
    result["gate_result"] = gate_cell(result, gate)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    env.gym.destroy_sim(env.sim)
    return result


if __name__ == "__main__":
    evaluate(_parse_args())

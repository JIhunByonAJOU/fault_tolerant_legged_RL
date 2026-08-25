"""Interactive Teacher243 FailureEnv viewer with readable degradation overlay."""

import argparse
import math
import multiprocessing as mp
import queue
import sys
from collections import deque
from pathlib import Path

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch
from isaacgym import gymapi
from isaacgym.torch_utils import get_euler_xyz
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.evaluation.teacher243_viewer_overlay import (
    ACTUAL_PATH_COLOR,
    NORMAL_COLOR,
    TARGET_PATH_COLOR,
    advance_reference,
    panel_rows,
    rigid_body_name_for_dof,
    severity_style,
    terrain_heights_at_xy,
)
from legged_gym.evaluation.teacher243_viewer_panel import run_panel
from legged_gym.utils import get_args, task_registry


SUPPORTED_TASKS = (
    "a1_official_wim_teacher243_failure",
    "a1_official_wim_teacher243_failure_fullrange",
)


def parse_args():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--checkpoint-path", required=True)
    own.add_argument("--trail-seconds", type=float, default=12.0)
    own.add_argument("--trail-sample-dt", type=float, default=0.10)
    own.add_argument("--degradation-mode", choices=("sampled", "fixed"), default="sampled")
    own.add_argument("--fixed-joint-index", type=int, default=0)
    own.add_argument("--fixed-degradation-rate", type=float, default=0.8)
    known, remaining = own.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _segments(points):
    array = np.asarray(points, dtype=np.float32)
    if len(array) < 2:
        return None
    return np.stack((array[:-1], array[1:]), axis=1).reshape(-1, 3)


def _draw_paths(env, targets, actuals, controls):
    env.gym.clear_lines(env.viewer)
    selected = int(controls["selected_env"])
    for index, handle in enumerate(env.envs):
        if controls["selected_only"] and index != selected:
            continue
        for enabled, points, color in (
            (controls["show_target"], targets[index], TARGET_PATH_COLOR),
            (controls["show_actual"], actuals[index], ACTUAL_PATH_COLOR),
        ):
            vertices = _segments(points) if enabled else None
            if vertices is None:
                continue
            colors = np.repeat(
                np.asarray([color], dtype=np.float32), len(vertices) // 2, axis=0
            )
            env.gym.add_lines(
                env.viewer, handle, len(vertices) // 2, vertices, colors
            )


def _body_handles(env):
    body_names = tuple(
        env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    )
    body_dict = env.gym.get_actor_rigid_body_dict(env.envs[0], env.actor_handles[0])
    handles = []
    for dof_name in env.dof_names:
        body_name = rigid_body_name_for_dof(dof_name, body_names)
        handles.append(int(body_dict[body_name]))
    return body_names, handles


def _apply_colors(env, joint_handles, enabled):
    white = gymapi.Vec3(*NORMAL_COLOR)
    joint_indices = env.degraded_joint_index.detach().cpu().tolist()
    rates = env.actuator_degradation.max(dim=1).values.detach().cpu().tolist()
    for index, (env_handle, actor_handle) in enumerate(
        zip(env.envs, env.actor_handles)
    ):
        body_count = env.gym.get_actor_rigid_body_count(env_handle, actor_handle)
        for body_index in range(body_count):
            env.gym.set_rigid_body_color(
                env_handle, actor_handle, body_index, gymapi.MESH_VISUAL, white
            )
        joint_index = int(joint_indices[index])
        if enabled and joint_index >= 0 and rates[index] > 0.0:
            color = gymapi.Vec3(*severity_style(rates[index]).rgb)
            env.gym.set_rigid_body_color(
                env_handle,
                actor_handle,
                joint_handles[joint_index],
                gymapi.MESH_VISUAL,
                color,
            )


def _degradation_snapshot(env):
    return (
        tuple(env.degraded_joint_index.detach().cpu().tolist()),
        tuple(
            round(float(value), 4)
            for value in env.actuator_degradation.max(dim=1).values.detach().cpu().tolist()
        ),
    )


def _send_status(status_queue, env, message):
    indices, rates = _degradation_snapshot(env)
    selected_strengths = []
    motor_strength = env.motor_strength_gt.detach().cpu()
    for env_index, joint_index in enumerate(indices):
        selected_strengths.append(
            1.0 if int(joint_index) < 0 else float(motor_strength[env_index, int(joint_index)])
        )
    payload = {
        "rows": panel_rows(
            indices,
            rates,
            list(env.dof_names),
            effective_motor_strengths=selected_strengths,
        ),
        "message": message,
    }
    try:
        status_queue.put_nowait(payload)
    except queue.Full:
        pass


def _drain_controls(control_queue, controls):
    clear = False
    focus = False
    quit_requested = False
    try:
        while True:
            update = control_queue.get_nowait()
            clear = clear or bool(update.pop("clear", False))
            focus = focus or bool(update.pop("focus", False))
            quit_requested = quit_requested or bool(update.pop("quit", False))
            controls.update(update)
    except queue.Empty:
        pass
    return clear, focus, quit_requested


def _initial_reference(env):
    # The terrain environments are created with a zero environment transform;
    # robot roots and viewer lines therefore share simulation/world coordinates.
    world_xy = env.root_states[:, :2].detach().cpu().numpy()
    yaw = get_euler_xyz(env.root_states[:, 3:7])[2].detach().cpu().numpy()
    return world_xy.astype(np.float64), yaw.astype(np.float64)


def _target_reference_z(env, world_xy):
    """Implicit base-height target above the terrain under each XY point."""
    if env.cfg.terrain.mesh_type == "plane":
        terrain_z = np.zeros(len(world_xy), dtype=np.float64)
    else:
        terrain_z = terrain_heights_at_xy(
            world_xy,
            env.height_samples.detach().cpu().numpy(),
            env.terrain.cfg.border_size,
            env.terrain.cfg.horizontal_scale,
            env.terrain.cfg.vertical_scale,
        )
    return terrain_z + float(env.cfg.rewards.base_height_target)


def _frame_all_robots(env):
    points = env.root_states[:, :2].detach().cpu().numpy()
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    span = max(float(x_max - x_min), float(y_max - y_min), 5.0)
    env.set_camera(
        [center_x - 0.75 * span, center_y - 1.10 * span, 0.80 * span],
        [center_x, center_y, 0.35],
    )


def _focus_robot(env, env_index):
    index = max(0, min(int(env_index), env.num_envs - 1))
    x, y, z = env.root_states[index, :3].detach().cpu().tolist()
    env.set_camera(
        [x - 2.2, y - 3.0, z + 1.55],
        [x, y, z + 0.15],
    )


def main():
    own, args = parse_args()
    if args.headless:
        raise ValueError("viewer requires graphics; omit --headless")
    if args.task not in SUPPORTED_TASKS:
        raise ValueError("viewer requires one of {}".format(SUPPORTED_TASKS))
    if args.num_envs is None:
        args.num_envs = 4
    if not 1 <= args.num_envs <= 8:
        raise ValueError("--num_envs must be between 1 and 8 for readable visualization")
    if own.trail_seconds <= 0.0 or own.trail_sample_dt <= 0.0:
        raise ValueError("trail durations must be positive")
    if not 0 <= own.fixed_joint_index < 12:
        raise ValueError("--fixed-joint-index must be in [0, 11]")
    if not 0.0 <= own.fixed_degradation_rate <= 1.0:
        raise ValueError("--fixed-degradation-rate must be in [0, 1]")
    checkpoint = Path(own.checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = args.num_envs
    # The training terrain is deliberately wide.  A compact viewer-only grid
    # keeps every robot large enough to read without changing training.
    terrain_cols = int(math.ceil(math.sqrt(args.num_envs)))
    terrain_rows = int(math.ceil(args.num_envs / terrain_cols))
    env_cfg.terrain.num_cols = max(1, terrain_cols)
    env_cfg.terrain.num_rows = max(1, terrain_rows)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(args.task, args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(checkpoint), load_optimizer=False)
    model = runner.alg.actor_critic
    model.eval()

    context = mp.get_context("spawn")
    control_queue = context.Queue(maxsize=16)
    status_queue = context.Queue(maxsize=4)
    panel = context.Process(
        target=run_panel,
        args=(control_queue, status_queue, env.num_envs),
        name="teacher243-viewer-panel",
    )
    panel.start()

    controls = {
        "show_target": True,
        "show_actual": True,
        "show_degradation": True,
        "selected_only": False,
        "selected_env": 0,
    }
    _, joint_handles = _body_handles(env)
    max_points = max(2, int(math.ceil(own.trail_seconds / own.trail_sample_dt)))
    env_ids = torch.arange(env.num_envs, device=env.device)
    obs, privileged = env.reset()
    # Bind both trails to the post-reset pose actually shown in the viewer.
    target_xy, target_yaw = _initial_reference(env)
    target_z = _target_reference_z(env, target_xy)
    actual_xyz = env.root_states[:, :3].detach().cpu().numpy()
    targets = [deque(maxlen=max_points) for _ in range(env.num_envs)]
    actuals = [deque(maxlen=max_points) for _ in range(env.num_envs)]
    for index in range(env.num_envs):
        targets[index].append((target_xy[index, 0], target_xy[index, 1], target_z[index]))
        actuals[index].append(tuple(float(value) for value in actual_xyz[index]))
    _frame_all_robots(env)
    sample_accumulator = 0.0
    last_snapshot = None
    last_highlight = None
    try:
        while panel.is_alive():
            clear, focus, quit_requested = _drain_controls(control_queue, controls)
            if quit_requested:
                break
            if focus:
                _focus_robot(env, controls["selected_env"])
            if clear:
                target_xy, target_yaw = _initial_reference(env)
                for collection in targets + actuals:
                    collection.clear()

            if own.degradation_mode == "fixed":
                env.set_actuator_degradation(
                    env_ids, own.fixed_joint_index, own.fixed_degradation_rate
                )
                env.compute_observations()
                obs = env.get_observations()
                privileged = env.get_privileged_observations()

            command = env.commands[:, :3].detach().cpu().numpy().copy()
            # commands[:, :2] are desired body-frame velocities.  Convert them
            # with measured yaw, not commands[:, 3] (desired heading), so the
            # green reference and blue motion use the same world frame.
            body_heading = (
                get_euler_xyz(env.root_states[:, 3:7])[2]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            target_xy, target_yaw = advance_reference(
                target_xy,
                target_yaw,
                command,
                env.dt,
                body_heading=body_heading,
            )
            # Environment buffers are updated in place; no_grad is correct here
            # while inference_mode would turn freshly assigned buffers into
            # immutable inference tensors across viewer iterations.
            with torch.no_grad():
                actions = model.act_inference(obs, privileged)
                obs, privileged, _, dones, _ = env.step(actions)

            done_indices = torch.nonzero(dones > 0, as_tuple=False).flatten().cpu().tolist()
            if done_indices:
                if own.degradation_mode == "fixed":
                    reset_ids = torch.as_tensor(
                        done_indices, dtype=torch.long, device=env.device
                    )
                    env.set_actuator_degradation(
                        reset_ids,
                        own.fixed_joint_index,
                        own.fixed_degradation_rate,
                    )
                    env.compute_observations()
                    obs = env.get_observations()
                    privileged = env.get_privileged_observations()
                reset_xy, reset_yaw = _initial_reference(env)
                reset_target_z = _target_reference_z(env, reset_xy)
                reset_actual_xyz = env.root_states[:, :3].detach().cpu().numpy()
                for index in done_indices:
                    target_xy[index] = reset_xy[index]
                    target_yaw[index] = reset_yaw[index]
                    targets[index].clear()
                    actuals[index].clear()
                    # Start both new-episode trails at the newly spawned pose;
                    # no segment from the previous robot lifetime can remain.
                    targets[index].append(
                        (reset_xy[index, 0], reset_xy[index, 1], reset_target_z[index])
                    )
                    actuals[index].append(tuple(float(value) for value in reset_actual_xyz[index]))

            sample_accumulator += env.dt
            if sample_accumulator + 1.0e-9 >= own.trail_sample_dt:
                sample_accumulator %= own.trail_sample_dt
                world_xyz = env.root_states[:, :3].detach().cpu().numpy()
                target_z = _target_reference_z(env, target_xy)
                for index in range(env.num_envs):
                    targets[index].append(
                        (target_xy[index, 0], target_xy[index, 1], target_z[index])
                    )
                    actuals[index].append(tuple(float(value) for value in world_xyz[index]))

            snapshot = _degradation_snapshot(env)
            if snapshot != last_snapshot or controls["show_degradation"] != last_highlight:
                _apply_colors(env, joint_handles, controls["show_degradation"])
                _send_status(
                    status_queue,
                    env,
                    "{} robots | {} degradation".format(
                        env.num_envs, own.degradation_mode
                    ),
                )
                last_snapshot = snapshot
                last_highlight = controls["show_degradation"]
            _draw_paths(env, targets, actuals, controls)
    finally:
        try:
            control_queue.put_nowait({"quit": True})
        except queue.Full:
            pass
        panel.join(timeout=2.0)
        if panel.is_alive():
            panel.terminate()
            panel.join(timeout=2.0)
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
            env.viewer = None
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()

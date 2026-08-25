"""Interactive multi-robot teacher viewer with target paths and bounded trails."""

from collections import deque

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.envs.a1_limping.schema import ActorObservationSlices
from legged_gym.evaluation.viewer_paths import target_polyline, viewer_command
from legged_gym.utils import get_args, task_registry


def _segments(points):
    return np.stack((points[:-1], points[1:]), axis=1).reshape(-1, 3).astype(np.float32)


def _draw_paths(env, targets, trails):
    env.gym.clear_lines(env.viewer)
    green = np.asarray([[0.1, 0.9, 0.1]], dtype=np.float32)
    blue = np.asarray([[0.1, 0.3, 1.0]], dtype=np.float32)
    for index, handle in enumerate(env.envs):
        target_vertices = _segments(targets[index])
        env.gym.add_lines(
            env.viewer,
            handle,
            len(target_vertices) // 2,
            target_vertices,
            np.repeat(green, len(target_vertices) // 2, axis=0),
        )
        if len(trails[index]) >= 2:
            trail_vertices = _segments(np.asarray(trails[index], dtype=np.float32))
            env.gym.add_lines(
                env.viewer,
                handle,
                len(trail_vertices) // 2,
                trail_vertices,
                np.repeat(blue, len(trail_vertices) // 2, axis=0),
            )


def _frame_all_robots(env, targets):
    origins = env.env_origins[:, :2].detach().cpu().numpy()
    points = []
    for index, target in enumerate(targets):
        points.append(target[:, :2] + origins[index])
    points = np.concatenate(points, axis=0)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, 6.0)
    env.set_camera(
        [center_x - 0.65 * span, center_y - 1.05 * span, 0.70 * span],
        [center_x, center_y, 0.3],
    )


def view(args):
    if args.headless:
        raise ValueError("view_teacher requires a viewer; omit --headless")
    if args.task != "a1_limping_base_v2":
        raise ValueError("view_teacher supports a1_limping_base_v2 only")
    if args.num_envs is None:
        args.num_envs = 8
    if not 1 <= args.num_envs <= 8:
        raise ValueError("viewer supports 1-8 robots and defaults to 8")
    seed = args.seed if args.seed is not None else 1
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.curriculum = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    actor_critic = runner.alg.actor_critic
    obs, privileged_obs = env.reset()
    targets = [target_polyline(i, seed=seed) for i in range(env.num_envs)]
    trails = [deque(maxlen=300) for _ in range(env.num_envs)]
    _frame_all_robots(env, targets)
    elapsed = 0.0
    with torch.inference_mode():
        while True:
            commands = torch.tensor(
                [viewer_command(i, elapsed, seed=seed) for i in range(env.num_envs)],
                dtype=env.commands.dtype,
                device=env.device,
            )
            env.commands[:, :3] = commands
            obs[:, ActorObservationSlices.COMMAND] = commands * env.commands_scale
            actions = actor_critic.act_inference(obs, privileged_obs)
            obs, privileged_obs, _, _, _ = env.step(actions)
            local_xy = (env.root_states[:, :2] - env.env_origins[:, :2]).detach().cpu().numpy()
            for index, xy in enumerate(local_xy):
                trails[index].append((float(xy[0]), float(xy[1]), 0.10))
            _draw_paths(env, targets, trails)
            elapsed += env.dt


if __name__ == "__main__":
    view(get_args())

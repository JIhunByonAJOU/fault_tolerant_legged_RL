"""Viewer-only target path generation; not imported by evaluation or training."""

import math

import numpy as np


def viewer_command(index, elapsed, seed=1):
    rng = np.random.RandomState(int(seed) + 7919 * int(index))
    jitter = rng.uniform(-0.02, 0.02)
    profiles = (
        (0.25 + jitter, 0.00, 0.00),
        (0.50 + jitter, 0.00, 0.00),
        (0.20 + jitter, 0.12, 0.00),
        (0.20 + jitter, -0.12, 0.00),
        (0.30 + jitter, 0.00, 0.16),
        (0.30 + jitter, 0.00, -0.16),
    )
    if index < len(profiles):
        return profiles[index]
    phase = int(elapsed // 5.0) % 4
    if index % 2 == 0:
        sequence = ((0.35, 0.0, 0.0), (0.32, 0.0, 0.12), (0.28, 0.08, 0.0), (0.32, 0.0, -0.12))
    else:
        sequence = ((0.30, 0.0, 0.0), (0.28, -0.08, 0.0), (0.32, 0.0, -0.12), (0.32, 0.0, 0.12))
    vx, vy, yaw = sequence[phase]
    return vx + jitter, vy, yaw


def target_polyline(index, duration=20.0, sample_dt=0.1, seed=1):
    points = []
    x = y = yaw = 0.0
    steps = int(round(duration / sample_dt))
    for step in range(steps + 1):
        points.append((x, y, 0.08))
        vx, vy, yaw_rate = viewer_command(index, step * sample_dt, seed=seed)
        world_vx = math.cos(yaw) * vx - math.sin(yaw) * vy
        world_vy = math.sin(yaw) * vx + math.cos(yaw) * vy
        x += world_vx * sample_dt
        y += world_vy * sample_dt
        yaw += yaw_rate * sample_dt
    return np.asarray(points, dtype=np.float32)

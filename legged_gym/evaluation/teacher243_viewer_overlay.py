"""Pure helpers for the Teacher243 FailureEnv viewer overlay."""

from dataclasses import dataclass

import numpy as np


TARGET_PATH_COLOR = (0.10, 0.90, 0.10)
ACTUAL_PATH_COLOR = (0.10, 0.30, 1.00)
NORMAL_COLOR = (1.00, 1.00, 1.00)


@dataclass(frozen=True)
class SeverityStyle:
    degradation: float
    name: str
    rgb: tuple
    hex_color: str


SEVERITY_STYLES = (
    SeverityStyle(0.0, "Normal", NORMAL_COLOR, "#FFFFFF"),
    SeverityStyle(0.2, "20% degraded", (1.00, 0.90, 0.00), "#FFE600"),
    SeverityStyle(0.4, "40% degraded", (1.00, 0.45, 0.00), "#FF7300"),
    SeverityStyle(0.6, "60% degraded", (0.95, 0.05, 0.05), "#F20D0D"),
    SeverityStyle(0.8, "80% degraded", (0.45, 0.18, 0.05), "#732E0D"),
    SeverityStyle(1.0, "100% degraded", (0.00, 0.00, 0.00), "#000000"),
)


def severity_style(degradation):
    """Return the nearest configured visual style for a degradation value."""
    value = float(degradation)
    if not 0.0 <= value <= 1.0:
        raise ValueError("degradation must be in [0, 1]")
    return min(SEVERITY_STYLES, key=lambda style: abs(style.degradation - value))


def rigid_body_name_for_dof(dof_name, body_names):
    """Map A1 joint names such as FR_calf_joint to their driven link."""
    bodies = tuple(body_names)
    stem = str(dof_name)
    if stem.endswith("_joint"):
        stem = stem[:-6]
    if stem in bodies:
        return stem
    matches = [name for name in bodies if name.startswith(stem) or stem.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("cannot map DOF {!r} to one rigid body".format(dof_name))


def advance_reference(target_xy, target_yaw, commands, dt, body_heading=None):
    """Integrate body-frame velocity commands into world coordinates.

    ``body_heading`` should be the robot's measured world yaw for the current
    control step.  The command is a body-frame velocity, not a pre-planned
    world path; using the desired heading here rotates the path incorrectly
    whenever heading tracking has error.
    """
    xy = np.asarray(target_xy, dtype=np.float64).copy()
    yaw = np.asarray(target_yaw, dtype=np.float64).copy()
    cmd = np.asarray(commands, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or cmd.shape != (xy.shape[0], 3):
        raise ValueError("target_xy must be Nx2 and commands must be Nx3")
    if yaw.shape != (xy.shape[0],):
        raise ValueError("target_yaw must be N")
    if body_heading is not None:
        measured = np.asarray(body_heading, dtype=np.float64)
        if measured.shape != yaw.shape:
            raise ValueError("body_heading must be N")
        yaw = measured.copy()
    c = np.cos(yaw)
    s = np.sin(yaw)
    xy[:, 0] += (c * cmd[:, 0] - s * cmd[:, 1]) * float(dt)
    xy[:, 1] += (s * cmd[:, 0] + c * cmd[:, 1]) * float(dt)
    if body_heading is None:
        yaw += cmd[:, 2] * float(dt)
    return xy, yaw


def terrain_heights_at_xy(
    world_xy, height_samples, border_size, horizontal_scale, vertical_scale
):
    """Sample the rendered terrain height under arbitrary world XY points.

    This mirrors ``LeggedRobot._get_heights`` so viewer references follow the
    same height field used by the environment.
    """
    xy = np.asarray(world_xy, dtype=np.float64)
    heights = np.asarray(height_samples)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("world_xy must be Nx2")
    if heights.ndim != 2 or min(heights.shape) < 2:
        raise ValueError("height_samples must be a 2D grid of at least 2x2")
    points = ((xy + float(border_size)) / float(horizontal_scale)).astype(np.int64)
    px = np.clip(points[:, 0], 0, heights.shape[0] - 2)
    py = np.clip(points[:, 1], 0, heights.shape[1] - 2)
    sampled = np.minimum(np.minimum(heights[px, py], heights[px + 1, py]), heights[px, py + 1])
    return sampled.astype(np.float64) * float(vertical_scale)


def panel_rows(
    joint_indices,
    degradation_rates,
    joint_names,
    effective_motor_strengths=None,
):
    if effective_motor_strengths is None:
        effective_motor_strengths = [1.0 - float(rate) for rate in degradation_rates]
    rows = []
    for env_index, (joint_index, rate, effective_strength) in enumerate(
        zip(joint_indices, degradation_rates, effective_motor_strengths)
    ):
        rate = float(rate)
        joint_index = int(joint_index)
        joint_name = "Normal" if joint_index < 0 or rate <= 0.0 else joint_names[joint_index]
        rows.append(
            {
                "env": env_index,
                "joint": joint_name,
                "degradation": rate,
                "remaining": 1.0 - rate,
                "effective_strength": float(effective_strength),
                "color": severity_style(rate).hex_color,
            }
        )
    return rows

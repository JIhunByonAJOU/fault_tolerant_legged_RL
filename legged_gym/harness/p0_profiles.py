"""Immutable named P0 contracts. The legacy teacher contract remains the default."""

from types import MappingProxyType


TEACHER45 = MappingProxyType({"name": "teacher45", "validator": "legacy"})
OFFICIAL_WIM_A1_ROUGH_V1 = MappingProxyType({
    "name": "official_wim_a1_rough_v1",
    "task": "a1_official_wim_rough",
    "num_envs": 64,
    "num_observations": 235,
    "num_privileged_obs": None,
    "num_actions": 12,
    "mesh_type": "trimesh",
    "measure_heights": True,
    "measured_points_x": 17,
    "measured_points_y": 11,
    "seed": 1,
    "num_steps_per_env": 24,
    "max_iterations": 2,
    "num_mini_batches": 4,
    "num_learning_epochs": 5,
    "save_interval": 1,
    "metric_iterations": (0, 1),
    "checkpoint_iterations": (0, 1, 2),
})
P0_PROFILES = MappingProxyType({
    "teacher45": TEACHER45,
    "official_wim_a1_rough_v1": OFFICIAL_WIM_A1_ROUGH_V1,
})

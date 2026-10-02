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
_COMPARISON_COMMON = {
    "validator": "comparison",
    "num_envs": 64,
    "seed": 1,
    "num_steps_per_env": 24,
    "max_iterations": 2,
    "num_mini_batches": 4,
    "num_learning_epochs": 5,
    "save_interval": 1,
    "step_sleep_ms": 30.0,
    "minibatch_sleep_ms": 20.0,
    "metric_iterations": (43000, 43001),
    "checkpoint_iterations": (43000, 43001, 43002),
    "source_sha256": "944a697abb30dfc8023e15544d0909acfcdaa4d8c4c0f930656847398f150635",
}
COMPARISON_B1 = MappingProxyType(dict(
    _COMPARISON_COMMON,
    name="comparison_b1",
    task="a1_official_wim_jt_history_free_onset",
    profile_id="b1_current_repeat_v1",
    comparison_name="current-repeat / temporal-history control",
    policy_class_name="JointTeacherStudentActorCritic",
    algorithm_class_name="TeacherPPO",
    runner_class_name="ComparisonJointTeacherStudentRunner",
    update_kind="ppo",
))
COMPARISON_B2 = MappingProxyType(dict(
    _COMPARISON_COMMON,
    name="comparison_b2",
    task="a1_official_wim_separate_student_onset",
    profile_id="b2_separate_student_v1",
    comparison_name="RMA-style internal two-stage control",
    policy_class_name="SeparateStudentActorCritic",
    algorithm_class_name="StudentDistillation",
    runner_class_name="SeparateStudentDistillationRunner",
    update_kind="supervised",
))
P0_PROFILES = MappingProxyType({
    "teacher45": TEACHER45,
    "official_wim_a1_rough_v1": OFFICIAL_WIM_A1_ROUGH_V1,
    "comparison_b1": COMPARISON_B1,
    "comparison_b2": COMPARISON_B2,
})

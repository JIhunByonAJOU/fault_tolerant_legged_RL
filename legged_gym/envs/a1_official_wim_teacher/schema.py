"""Tensor contract for the conservative WIM-235 + teacher-latent-8 task."""


WIM_OBSERVATION_DIM = 235
PRIVILEGED_OBSERVATION_DIM = 45
TEACHER_LATENT_DIM = 8
ACTOR_INPUT_DIM = WIM_OBSERVATION_DIM + TEACHER_LATENT_DIM


class PrivilegedObservationSlices:
    FRICTION = slice(0, 1)
    ADDED_BASE_MASS = slice(1, 2)
    MOTOR_STRENGTH = slice(2, 14)
    KP_SCALE = slice(14, 26)
    KD_SCALE = slice(26, 38)
    CLEAN_BASE_LINEAR_VELOCITY = slice(38, 41)
    CLEAN_BASE_ANGULAR_VELOCITY = slice(41, 44)
    FAILURE_FLAG = slice(44, 45)


def schema_manifest():
    return {
        "actor_observation_dim": WIM_OBSERVATION_DIM,
        "privileged_observation_dim": PRIVILEGED_OBSERVATION_DIM,
        "teacher_latent_dim": TEACHER_LATENT_DIM,
        "actor_input_dim": ACTOR_INPUT_DIM,
        "actor_input": "official WIM obs235 concatenated with teacher latent8",
        "terrain_contract": "official WIM 187 height samples remain directly observable",
        "privileged": {
            "applied_friction_scalar": [0, 1],
            "applied_added_payload_kg": [1, 2],
            "applied_motor_strength": [2, 14],
            "applied_kp_scale": [14, 26],
            "applied_kd_scale": [26, 38],
            "clean_body_frame_base_linear_velocity": [38, 41],
            "clean_body_frame_base_angular_velocity": [41, 44],
            "failure_flag_reserved_for_failure_env": [44, 45],
        },
    }

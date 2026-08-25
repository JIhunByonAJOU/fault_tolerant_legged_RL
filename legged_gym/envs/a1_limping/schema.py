"""Exact plane BaseEnv tensor layouts for the privileged A1 teacher."""


class ActorObservationSlices:
    DOF_POSITION = slice(0, 12)
    DOF_VELOCITY = slice(12, 24)
    ROLL_PITCH = slice(24, 26)
    FOOT_CONTACTS = slice(26, 30)
    PREVIOUS_ACTION = slice(30, 42)
    COMMAND = slice(42, 45)


class PrivilegedObservationSlices:
    FRICTION = slice(0, 1)
    ADDED_BASE_MASS = slice(1, 2)
    MOTOR_STRENGTH = slice(2, 14)
    KP_SCALE = slice(14, 26)
    KD_SCALE = slice(26, 38)
    CLEAN_BASE_LINEAR_VELOCITY = slice(38, 41)
    CLEAN_BASE_ANGULAR_VELOCITY = slice(41, 44)
    FAILURE_FLAG = slice(44, 45)


ACTOR_OBSERVATION_DIM = 45
COMMAND_CONDITIONED_ACTOR_OBSERVATION_DIM = ACTOR_OBSERVATION_DIM
PRIVILEGED_OBSERVATION_DIM = 45
TEACHER_LATENT_DIM = 8


def schema_manifest(actor_observation_dim=ACTOR_OBSERVATION_DIM):
    if actor_observation_dim != ACTOR_OBSERVATION_DIM:
        raise ValueError(
            "Plane A1 actor observations must have dimension {}, got {}".format(
                ACTOR_OBSERVATION_DIM, actor_observation_dim
            )
        )

    input_dim = ACTOR_OBSERVATION_DIM + TEACHER_LATENT_DIM
    return {
        "actor_observation_dim": ACTOR_OBSERVATION_DIM,
        "privileged_observation_dim": PRIVILEGED_OBSERVATION_DIM,
        "teacher_latent_dim": TEACHER_LATENT_DIM,
        "actor_input_dim": input_dim,
        "critic_input_dim": input_dim,
        "actor": {
            "q_default_centered": [0, 12],
            "qdot": [12, 24],
            "roll_pitch": [24, 26],
            "foot_contacts_fl_fr_rl_rr": [26, 30],
            "previous_raw_policy_action": [30, 42],
            "command_vx_vy_yaw": [42, 45],
        },
        "privileged": {
            "applied_friction_scalar": [0, 1],
            "applied_added_payload_kg": [1, 2],
            "applied_motor_strength": [2, 14],
            "applied_kp_scale": [14, 26],
            "applied_kd_scale": [26, 38],
            "clean_body_frame_base_linear_velocity": [38, 41],
            "clean_body_frame_base_angular_velocity": [41, 44],
            "reserved_failure_flag": [44, 45],
        },
    }

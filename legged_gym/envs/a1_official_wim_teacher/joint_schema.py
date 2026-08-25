"""Tensor contract for the WIM/Saving joint teacher-student task."""

CURRENT_OBSERVATION_DIM = 235
HISTORY_FRAME_DIM = 48
HISTORY_LENGTH = 50
HISTORY_OBSERVATION_DIM = HISTORY_FRAME_DIM * HISTORY_LENGTH
JOINT_OBSERVATION_DIM = CURRENT_OBSERVATION_DIM + HISTORY_OBSERVATION_DIM
PRIVILEGED_OBSERVATION_DIM = 45
LATENT_DIM = 8
POLICY_INPUT_DIM = CURRENT_OBSERVATION_DIM + LATENT_DIM


def schema_manifest():
    return {
        "current_observation_dim": CURRENT_OBSERVATION_DIM,
        "history_frame_dim": HISTORY_FRAME_DIM,
        "history_length": HISTORY_LENGTH,
        "history_observation_dim": HISTORY_OBSERVATION_DIM,
        "stored_actor_observation_dim": JOINT_OBSERVATION_DIM,
        "privileged_observation_dim": PRIVILEGED_OBSERVATION_DIM,
        "teacher_latent_dim": LATENT_DIM,
        "student_latent_dim": LATENT_DIM,
        "policy_input_dim": POLICY_INPUT_DIM,
        "policy_input": "current WIM obs235 + fused latent8",
        "student_input": "50 frames of the first 48 WIM observation values",
    }

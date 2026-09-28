"""Joint teacher-student configuration built from the selected TF lineage."""

from .a1_official_wim_teacher_failure_config import (
    A1OfficialWimTeacher243FailureFullRangeCfg,
    A1OfficialWimTeacher243FailureFullRangeCfgPPO,
)
from .joint_schema import JOINT_OBSERVATION_DIM


class A1OfficialWimJointFailureOnsetCfg(
    A1OfficialWimTeacher243FailureFullRangeCfg
):
    class env(A1OfficialWimTeacher243FailureFullRangeCfg.env):
        task_name = "a1_official_wim_jt_failure_fullrange_onset"
        num_observations = JOINT_OBSERVATION_DIM

    class domain_rand(A1OfficialWimTeacher243FailureFullRangeCfg.domain_rand):
        # Saving specifies random onset but not its bounds. This project choice
        # matches the already evaluated P2.5 onset window.
        failure_onset_time_range_s = [2.0, 10.0]


class A1OfficialWimJointFailureOnsetCfgPPO(
    A1OfficialWimTeacher243FailureFullRangeCfgPPO
):
    runner_class_name = "JointTeacherStudentRunner"

    class policy(A1OfficialWimTeacher243FailureFullRangeCfgPPO.policy):
        history_frame_dim = 48
        history_length = 50
        student_embedding_dim = 32
        student_latent_dim = 8
        joint_schedule_iterations = 10000

    class runner(A1OfficialWimTeacher243FailureFullRangeCfgPPO.runner):
        policy_class_name = "JointTeacherStudentActorCritic"
        algorithm_class_name = "TeacherPPO"
        experiment_name = "jt_wim243_failure_fullrange_onset"
        run_name = ""
        max_iterations = 10000
        save_interval = 500
        resume = False
        checkpoint = -1
        load_run = -1


class A1OfficialWimJointBetaFloorCfg(A1OfficialWimJointFailureOnsetCfg):
    """Same environment as canonical JT; only the training profile changes."""

    class env(A1OfficialWimJointFailureOnsetCfg.env):
        task_name = "a1_official_wim_jt_beta_floor_onset"


class A1OfficialWimJointBetaFloorCfgPPO(A1OfficialWimJointFailureOnsetCfgPPO):
    """One-variable ablation: retain a small adaptation loss at alpha=1."""

    class policy(A1OfficialWimJointFailureOnsetCfgPPO.policy):
        adaptation_beta_floor = 0.1

    class runner(A1OfficialWimJointFailureOnsetCfgPPO.runner):
        experiment_name = "jt_wim243_beta_floor_onset"


class A1OfficialWimFrozenTeacherStudentCfg(A1OfficialWimJointFailureOnsetCfg):
    """Same random-onset environment for the frozen-TF student baseline."""

    class env(A1OfficialWimJointFailureOnsetCfg.env):
        task_name = "a1_official_wim_frozen_tf_student_onset"


class A1OfficialWimFrozenTeacherStudentCfgPPO(A1OfficialWimJointFailureOnsetCfgPPO):
    class policy(A1OfficialWimJointFailureOnsetCfgPPO.policy):
        teacher_only_iterations = 2000
        student_transition_iterations = 8000

    class runner(A1OfficialWimJointFailureOnsetCfgPPO.runner):
        policy_class_name = "FrozenTeacherStudentActorCritic"
        experiment_name = "frozen_tf_student_wim243_onset"

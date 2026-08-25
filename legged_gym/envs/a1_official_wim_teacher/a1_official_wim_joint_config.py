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

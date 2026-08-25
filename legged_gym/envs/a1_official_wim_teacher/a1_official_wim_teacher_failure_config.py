"""FailureEnv configuration for the WIM-235 plus privileged-latent teacher."""

from .a1_official_wim_teacher_config import (
    A1OfficialWimTeacher243Cfg,
    A1OfficialWimTeacher243CfgPPO,
)


class A1OfficialWimTeacher243FailureCfg(A1OfficialWimTeacher243Cfg):
    class env(A1OfficialWimTeacher243Cfg.env):
        task_name = "a1_official_wim_teacher243_failure"

    class domain_rand(A1OfficialWimTeacher243Cfg.domain_rand):
        # ADAPT convention: d=0 is intact; applied torque is desired*(1-d).
        # Uniform sampling gives 20% intact episodes and 20% at each severity.
        actuator_degradation_levels = [0.0, 0.2, 0.4, 0.6, 0.8]


class A1OfficialWimTeacher243FailureCfgPPO(A1OfficialWimTeacher243CfgPPO):
    class runner(A1OfficialWimTeacher243CfgPPO.runner):
        experiment_name = "official_wim_teacher243_failure"
        run_name = ""
        resume = False
        checkpoint = -1
        load_run = -1


class A1OfficialWimTeacher243FailureFullRangeCfg(
    A1OfficialWimTeacher243FailureCfg
):
    """Separate 0.0--1.0 degradation task; preserves the original task."""

    class env(A1OfficialWimTeacher243FailureCfg.env):
        task_name = "a1_official_wim_teacher243_failure_fullrange"

    class domain_rand(A1OfficialWimTeacher243FailureCfg.domain_rand):
        # Uniform per episode: intact or one random actuator at the chosen d.
        # d=1.0 removes commanded torque but does not lock the joint position.
        actuator_degradation_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class A1OfficialWimTeacher243FailureFullRangeCfgPPO(
    A1OfficialWimTeacher243FailureCfgPPO
):
    class runner(A1OfficialWimTeacher243FailureCfgPPO.runner):
        experiment_name = "official_wim_teacher243_failure_fullrange"
        run_name = ""
        resume = False
        checkpoint = -1
        load_run = -1

"""Isolated from-scratch control for the FullRange FailureEnv Teacher243."""

from .a1_official_wim_teacher_failure_config import (
    A1OfficialWimTeacher243FailureFullRangeCfg,
    A1OfficialWimTeacher243FailureFullRangeCfgPPO,
)


class A1OfficialWimTeacher243FailureFullRangeFromScratchCfg(
    A1OfficialWimTeacher243FailureFullRangeCfg
):
    """Same FullRange physics, registered separately from warm-start runs."""

    class env(A1OfficialWimTeacher243FailureFullRangeCfg.env):
        task_name = "a1_official_wim_teacher243_failure_fullrange_fromscratch"


class A1OfficialWimTeacher243FailureFullRangeFromScratchCfgPPO(
    A1OfficialWimTeacher243FailureFullRangeCfgPPO
):
    """Randomly initialized 50k ceiling with an isolated log namespace."""

    class runner(A1OfficialWimTeacher243FailureFullRangeCfgPPO.runner):
        experiment_name = "official_wim_teacher243_failure_fullrange_fromscratch"
        run_name = "seed1-50000iter"
        max_iterations = 50000
        save_interval = 500
        resume = False
        checkpoint = -1
        load_run = -1

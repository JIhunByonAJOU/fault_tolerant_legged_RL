"""Inheritance-only managed profile for the official stock A1 rough baseline."""

from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO


class A1OfficialWimRoughCfg(A1RoughCfg):
    """No scientific overrides are permitted in this profile."""


class A1OfficialWimRoughCfgPPO(A1RoughCfgPPO):
    runner_class_name = "OfficialWimOnPolicyRunner"

    class runner(A1RoughCfgPPO.runner):
        experiment_name = "official_wim_a1_rough"
        run_name = ""

"""Conservative 243-D teacher built on the successful official WIM task."""

from legged_gym.envs.a1_official_wim import (
    A1OfficialWimRoughCfg,
    A1OfficialWimRoughCfgPPO,
)

from .schema import PRIVILEGED_OBSERVATION_DIM, WIM_OBSERVATION_DIM


class A1OfficialWimTeacher243Cfg(A1OfficialWimRoughCfg):
    class env(A1OfficialWimRoughCfg.env):
        task_name = "a1_official_wim_teacher243"
        num_observations = WIM_OBSERVATION_DIM
        num_privileged_obs = PRIVILEGED_OBSERVATION_DIM

    class domain_rand(A1OfficialWimRoughCfg.domain_rand):
        # Friction and pushes are inherited from official WIM.  The remaining
        # mild variations make the privileged latent identifiable in BaseEnv.
        randomize_base_mass = True
        added_mass_range = [0.0, 2.0]
        motor_strength_range = [0.9, 1.1]
        kp_scale_range = [0.95, 1.05]
        kd_scale_range = [0.9, 1.1]


class A1OfficialWimTeacher243CfgPPO(A1OfficialWimRoughCfgPPO):
    runner_class_name = "TeacherOnPolicyRunner"

    class policy(A1OfficialWimRoughCfgPPO.policy):
        init_noise_std = 1.0
        teacher_hidden_dims = [512, 256, 128]
        teacher_latent_dim = 8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(A1OfficialWimRoughCfgPPO.algorithm):
        min_learning_rate = 1.0e-5
        max_learning_rate = 1.0e-2

    class runner(A1OfficialWimRoughCfgPPO.runner):
        policy_class_name = "OfficialWimTeacherActorCritic"
        algorithm_class_name = "TeacherPPO"
        experiment_name = "official_wim_teacher243"
        run_name = ""
        resume = False
        checkpoint = -1
        load_run = -1

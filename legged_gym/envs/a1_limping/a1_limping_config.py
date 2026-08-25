import math

from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO

from .schema import ACTOR_OBSERVATION_DIM, PRIVILEGED_OBSERVATION_DIM


EXPERIMENT_NAME = "Official initial WIM A1 + plane Saving teacher45 baseline"


class A1LimpingBaseCfg(A1RoughCfg):
    """WIM-flat observations/MDP composed with official initial A1 mechanics."""

    class env(A1RoughCfg.env):
        task_name = "a1_limping_base"
        num_envs = 4096
        num_observations = ACTOR_OBSERVATION_DIM
        num_privileged_obs = PRIVILEGED_OBSERVATION_DIM
        num_actions = 12
        episode_length_s = 20

    class terrain(A1RoughCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class commands(A1RoughCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges(A1RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-math.pi, math.pi]

    class domain_rand(A1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [0.0, 6.0]
        motor_strength_range = [0.9, 1.1]
        kp_scale_range = [50.0 / 55.0, 60.0 / 55.0]
        kd_scale_range = [0.4 / 0.6, 0.8 / 0.6]
        push_robots = True
        push_interval_s = 10.0
        max_push_vel_xy = 1.0

    class noise(A1RoughCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(A1RoughCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.01
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class normalization(A1RoughCfg.normalization):
        clip_actions = 1.0

    class rewards(A1RoughCfg.rewards):
        # Proven active in ae614 LeggedRobot.compute_reward for A1RoughCfg.
        only_positive_rewards = True
        tracking_sigma = 0.25

        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_vel = 0.0
            torques = -1.0e-5
            action_rate = -0.01
            collision = -1.0
            feet_air_time = 1.0

    class privileged:
        foot_contact_force_threshold = 1.0

        failure_flag = 0.0


class A1LimpingBaseCfgPPO(A1RoughCfgPPO):
    runner_class_name = "TeacherOnPolicyRunner"

    class policy(A1RoughCfgPPO.policy):
        init_noise_std = 1.0
        teacher_hidden_dims = [512, 256, 128]
        teacher_latent_dim = 8
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(A1RoughCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        desired_kl = 0.01
        min_learning_rate = 1.0e-5
        max_learning_rate = 1.0e-2
        gamma = 0.99
        lam = 0.95
        max_grad_norm = 1.0

    class runner(A1RoughCfgPPO.runner):
        policy_class_name = "TeacherActorCritic"
        algorithm_class_name = "TeacherPPO"
        experiment_name = EXPERIMENT_NAME
        run_name = ""
        num_steps_per_env = 24
        max_iterations = 1500
        resume = False
        checkpoint = -1
        load_run = -1


class A1LimpingBaseWimCfg(A1LimpingBaseCfg):
    class env(A1LimpingBaseCfg.env):
        task_name = "a1_limping_base_wim"


class A1LimpingBaseWimCfgPPO(A1LimpingBaseCfgPPO):
    pass


class A1LimpingBaseV2Cfg(A1LimpingBaseCfg):
    """Stable task key for the exact WIM/Saving P1 baseline."""

    class env(A1LimpingBaseCfg.env):
        task_name = "a1_limping_base_v2"


class A1LimpingBaseV2CfgPPO(A1LimpingBaseCfgPPO):
    pass

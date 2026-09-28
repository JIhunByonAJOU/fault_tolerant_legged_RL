"""Saving-style fused teacher/student policy on the selected WIM teacher."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from legged_gym.envs.a1_official_wim_teacher.joint_schema import (
    CURRENT_OBSERVATION_DIM,
    HISTORY_FRAME_DIM,
    HISTORY_LENGTH,
    JOINT_OBSERVATION_DIM,
    LATENT_DIM,
    POLICY_INPUT_DIM,
    PRIVILEGED_OBSERVATION_DIM,
)
from .official_wim_teacher_actor_critic import _mlp


class StudentHistoryEncoder(nn.Module):
    """RMA-lineage 50-frame 1-D CNN, projected to latent8."""

    def __init__(self):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Linear(HISTORY_FRAME_DIM, 32), nn.ELU()
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ELU(),
        )
        self.projection = nn.Linear(32 * 3, LATENT_DIM)

    def forward(self, history):
        if history.shape[-2:] != (HISTORY_LENGTH, HISTORY_FRAME_DIM):
            raise RuntimeError("student history must be 50x48")
        embedded = self.frame_encoder(history).transpose(1, 2)
        return self.projection(self.temporal(embedded).flatten(start_dim=1))


class JointTeacherStudentActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_privileged_obs,
        num_actions,
        teacher_hidden_dims=(512, 256, 128),
        teacher_latent_dim=8,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        history_frame_dim=48,
        history_length=50,
        student_embedding_dim=32,
        student_latent_dim=8,
        joint_schedule_iterations=10000,
        adaptation_beta_floor=0.0,
        **kwargs
    ):
        super().__init__()
        if kwargs:
            print("JointTeacherStudentActorCritic ignored: {}".format(sorted(kwargs)))
        expected = (
            num_actor_obs == JOINT_OBSERVATION_DIM
            and num_privileged_obs == PRIVILEGED_OBSERVATION_DIM
            and teacher_latent_dim == LATENT_DIM
            and student_latent_dim == LATENT_DIM
            and history_frame_dim == HISTORY_FRAME_DIM
            and history_length == HISTORY_LENGTH
            and student_embedding_dim == 32
        )
        if not expected:
            raise ValueError("JT tensor contract mismatch")
        self.num_actor_obs = num_actor_obs
        self.num_privileged_obs = num_privileged_obs
        self.teacher_latent_dim = LATENT_DIM
        self.teacher_encoder = _mlp(
            PRIVILEGED_OBSERVATION_DIM,
            teacher_hidden_dims,
            LATENT_DIM,
            activation,
        )
        self.student_encoder = StudentHistoryEncoder()
        self.actor = _mlp(POLICY_INPUT_DIM, actor_hidden_dims, num_actions, activation)
        self.critic = _mlp(POLICY_INPUT_DIM, critic_hidden_dims, 1, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.joint_schedule_iterations = int(joint_schedule_iterations)
        self.adaptation_beta_floor = float(adaptation_beta_floor)
        if not 0.0 <= self.adaptation_beta_floor < 1.0:
            raise ValueError("adaptation_beta_floor must be in [0, 1)")
        self.schedule_origin_iteration = 0
        self.adaptation_alpha = 0.0
        self.adaptation_beta = 1.0
        Normal.set_default_validate_args = False

    def forward(self):
        raise NotImplementedError

    def reset(self, dones=None):
        return None

    def set_schedule_origin(self, iteration):
        self.schedule_origin_iteration = int(iteration)

    def set_training_iteration(self, iteration):
        progress = (int(iteration) - self.schedule_origin_iteration) / max(
            self.joint_schedule_iterations, 1
        )
        self.adaptation_alpha = min(max(progress, 0.0), 1.0)
        self.adaptation_beta = self.adaptation_beta_floor + (
            1.0 - self.adaptation_beta_floor
        ) * (1.0 - self.adaptation_alpha)

    def _split(self, observations):
        if observations.shape[-1] != JOINT_OBSERVATION_DIM:
            raise RuntimeError("JT stored observation width changed")
        current = observations[:, :CURRENT_OBSERVATION_DIM]
        history = observations[:, CURRENT_OBSERVATION_DIM:].reshape(
            -1, HISTORY_LENGTH, HISTORY_FRAME_DIM
        )
        return current, history

    def encode_privileged(self, privileged_observations):
        return self.teacher_encoder(privileged_observations)

    def encode_history(self, observations):
        _, history = self._split(observations)
        return self.student_encoder(history)

    def fused_latent(self, observations, privileged_observations):
        teacher = self.encode_privileged(privileged_observations)
        student = self.encode_history(observations)
        alpha = self.adaptation_alpha
        return alpha * student + (1.0 - alpha) * teacher

    def adaptation_loss(self, observations, privileged_observations):
        target = self.encode_privileged(privileged_observations).detach()
        predicted = self.encode_history(observations)
        return torch.linalg.vector_norm(predicted - target, dim=-1).mean()

    def _policy_input(self, observations, privileged_observations):
        current, _ = self._split(observations)
        return torch.cat(
            (current, self.fused_latent(observations, privileged_observations)),
            dim=-1,
        )

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, privileged_observations):
        mean = self.actor(self._policy_input(observations, privileged_observations))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act_with_raw(self, observations, privileged_observations):
        self.update_distribution(observations, privileged_observations)
        actions = self.distribution.sample()
        return actions, actions

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations, privileged_observations, **kwargs):
        return self.critic(self._policy_input(observations, privileged_observations))

    def act_inference(self, observations, privileged_observations):
        return self.actor(self._policy_input(observations, privileged_observations))

    def act_inference_with_latent(self, observations, latent):
        current, _ = self._split(observations)
        return self.actor(torch.cat((current, latent), dim=-1))

    def act_inference_student(self, observations):
        current, _ = self._split(observations)
        latent = self.encode_history(observations)
        return self.actor(torch.cat((current, latent), dim=-1))


class FrozenTeacherStudentActorCritic(JointTeacherStudentActorCritic):
    """Diagnostic baseline: keep the TF policy fixed while learning its student."""

    def __init__(
        self,
        *args,
        teacher_only_iterations=2000,
        student_transition_iterations=8000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if teacher_only_iterations < 0 or student_transition_iterations <= 0:
            raise ValueError("invalid frozen-teacher transition schedule")
        self.teacher_only_iterations = int(teacher_only_iterations)
        self.student_transition_iterations = int(student_transition_iterations)
        self.teacher_encoder.requires_grad_(False)
        self.actor.requires_grad_(False)
        self.std.requires_grad_(False)

    def set_training_iteration(self, iteration):
        elapsed = int(iteration) - self.schedule_origin_iteration
        progress = (elapsed - self.teacher_only_iterations) / self.student_transition_iterations
        self.adaptation_alpha = min(max(progress, 0.0), 1.0)
        self.adaptation_beta = 1.0

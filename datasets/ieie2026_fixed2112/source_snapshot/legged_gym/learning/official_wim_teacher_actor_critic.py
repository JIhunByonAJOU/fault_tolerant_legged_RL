"""Saving-style teacher encoder attached to the successful WIM 235-D policy."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from legged_gym.envs.a1_official_wim_teacher.schema import (
    ACTOR_INPUT_DIM,
    PRIVILEGED_OBSERVATION_DIM,
    TEACHER_LATENT_DIM,
    WIM_OBSERVATION_DIM,
)


def _activation(name):
    values = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in values:
        raise ValueError("Unsupported activation: {}".format(name))
    return values[name]


def _mlp(input_dim, hidden_dims, output_dim, activation):
    layers = []
    previous = input_dim
    activation_class = _activation(activation)
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, hidden), activation_class()))
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class OfficialWimTeacherActorCritic(nn.Module):
    """Actor input is exactly WIM obs235 concatenated with teacher latent8."""

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
        **kwargs
    ):
        super().__init__()
        if kwargs:
            print("OfficialWimTeacherActorCritic ignored: {}".format(sorted(kwargs)))
        if num_actor_obs != WIM_OBSERVATION_DIM:
            raise ValueError("teacher243 requires WIM obs235")
        if num_privileged_obs != PRIVILEGED_OBSERVATION_DIM:
            raise ValueError("teacher243 requires privileged45")
        if teacher_latent_dim != TEACHER_LATENT_DIM:
            raise ValueError("teacher243 requires latent8")
        if list(teacher_hidden_dims) != [512, 256, 128]:
            raise ValueError("teacher encoder must be [512, 256, 128]")
        if list(actor_hidden_dims) != [512, 256, 128]:
            raise ValueError("teacher243 actor must preserve WIM [512, 256, 128]")
        if list(critic_hidden_dims) != [512, 256, 128]:
            raise ValueError("teacher243 critic must preserve WIM [512, 256, 128]")

        self.num_actor_obs = num_actor_obs
        self.num_privileged_obs = num_privileged_obs
        self.teacher_latent_dim = teacher_latent_dim
        self.teacher_encoder = _mlp(
            num_privileged_obs,
            teacher_hidden_dims,
            teacher_latent_dim,
            activation,
        )
        self.actor = _mlp(
            ACTOR_INPUT_DIM, actor_hidden_dims, num_actions, activation
        )
        self.critic = _mlp(
            ACTOR_INPUT_DIM, critic_hidden_dims, 1, activation
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print("Teacher encoder: {}".format(self.teacher_encoder))
        print("Teacher243 actor: {}".format(self.actor))
        print("Teacher243 critic: {}".format(self.critic))

    def forward(self):
        raise NotImplementedError

    def reset(self, dones=None):
        return None

    def encode_privileged(self, privileged_observations):
        if privileged_observations.shape[-1] != self.num_privileged_obs:
            raise RuntimeError("teacher243 privileged width changed")
        return self.teacher_encoder(privileged_observations)

    def policy_input_with_latent(self, observations, latent):
        if observations.shape[-1] != self.num_actor_obs:
            raise RuntimeError("teacher243 WIM observation width changed")
        if latent.shape[-1] != self.teacher_latent_dim:
            raise RuntimeError("teacher243 latent width changed")
        return torch.cat((observations, latent), dim=-1)

    def _policy_input(self, observations, privileged_observations):
        return self.policy_input_with_latent(
            observations, self.encode_privileged(privileged_observations)
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

    def act(self, observations, privileged_observations, **kwargs):
        self.update_distribution(observations, privileged_observations)
        return self.distribution.sample()

    def act_with_raw(self, observations, privileged_observations):
        actions = self.act(observations, privileged_observations)
        return actions, actions

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_raw_actions_log_prob(self, raw_actions):
        return self.get_actions_log_prob(raw_actions)

    def evaluate(self, observations, privileged_observations, **kwargs):
        return self.critic(self._policy_input(observations, privileged_observations))

    def act_inference(self, observations, privileged_observations):
        return self.actor(self._policy_input(observations, privileged_observations))

    def act_inference_raw(self, observations, privileged_observations):
        return self.act_inference(observations, privileged_observations)

    def act_inference_with_latent(self, observations, latent):
        return self.actor(self.policy_input_with_latent(observations, latent))

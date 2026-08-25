import torch
import torch.nn as nn
from torch.distributions import Normal

from legged_gym.envs.a1_limping.schema import (
    ACTOR_OBSERVATION_DIM,
    PRIVILEGED_OBSERVATION_DIM,
    TEACHER_LATENT_DIM,
)


def _activation(name):
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError("Unsupported activation: {}".format(name))
    return activations[name]


def _mlp(input_dim, hidden_dims, output_dim, activation):
    layers = []
    previous_dim = input_dim
    activation_class = _activation(activation)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(activation_class())
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class TeacherActorCritic(nn.Module):
    """Privileged teacher from Saving the Limping without a student path."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_privileged_obs,
        num_actions,
        teacher_hidden_dims=(512, 256, 128),
        teacher_latent_dim=8,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        **kwargs
    ):
        super().__init__()
        if kwargs:
            print(
                "TeacherActorCritic ignored unexpected arguments: {}".format(
                    sorted(kwargs.keys())
                )
            )

        if num_actor_obs != ACTOR_OBSERVATION_DIM:
            raise ValueError(
                "TeacherActorCritic requires WIM-flat obs{}, got {}".format(
                    ACTOR_OBSERVATION_DIM, num_actor_obs
                )
            )
        if teacher_latent_dim != TEACHER_LATENT_DIM:
            raise ValueError(
                "TeacherActorCritic requires latent{}, got {}".format(
                    TEACHER_LATENT_DIM, teacher_latent_dim
                )
            )
        if num_privileged_obs != PRIVILEGED_OBSERVATION_DIM:
            raise ValueError(
                "TeacherActorCritic requires privileged{}, got {}".format(
                    PRIVILEGED_OBSERVATION_DIM, num_privileged_obs
                )
            )
        if list(teacher_hidden_dims) != [512, 256, 128]:
            raise ValueError("Teacher encoder hidden dimensions must be [512, 256, 128]")
        if list(actor_hidden_dims) != [256, 128]:
            raise ValueError("Teacher actor hidden dimensions must be [256, 128]")

        self.num_actor_obs = num_actor_obs
        self.num_privileged_obs = num_privileged_obs
        self.teacher_latent_dim = teacher_latent_dim

        self.teacher_encoder = _mlp(
            num_privileged_obs,
            teacher_hidden_dims,
            teacher_latent_dim,
            activation,
        )
        policy_input_dim = num_actor_obs + teacher_latent_dim
        if policy_input_dim != 53:
            raise ValueError("Teacher actor input must be obs45 + latent8 = 53")
        self.actor = _mlp(
            policy_input_dim, actor_hidden_dims, num_actions, activation
        )
        self.critic = _mlp(
            policy_input_dim, critic_hidden_dims, 1, activation
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print("Teacher encoder: {}".format(self.teacher_encoder))
        print("Teacher actor: {}".format(self.actor))
        print("Teacher critic: {}".format(self.critic))

    def forward(self):
        raise NotImplementedError

    def reset(self, dones=None):
        return None

    def encode_privileged(self, privileged_observations):
        return self.teacher_encoder(privileged_observations)

    def _policy_input(self, observations, privileged_observations):
        if observations.shape[-1] != self.num_actor_obs:
            raise RuntimeError("Actor observation width changed at runtime")
        if privileged_observations.shape[-1] != self.num_privileged_obs:
            raise RuntimeError("Privileged observation width changed at runtime")
        latent = self.encode_privileged(privileged_observations)
        policy_input = torch.cat((observations, latent), dim=-1)
        if policy_input.shape[-1] != 53:
            raise RuntimeError("Teacher policy input width must remain 53")
        return policy_input

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        # Official WIM PPO entropy is evaluated in the raw Gaussian coordinate.
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, privileged_observations):
        mean = self.actor(self._policy_input(observations, privileged_observations))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, privileged_observations, **kwargs):
        raw_actions, _ = self.act_with_raw(observations, privileged_observations)
        return raw_actions

    def act_with_raw(self, observations, privileged_observations):
        """Return one raw Gaussian coordinate for environment and storage."""
        self.update_distribution(observations, privileged_observations)
        raw_actions = self.distribution.sample()
        return raw_actions, raw_actions

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_raw_actions_log_prob(self, raw_actions):
        """Compatibility alias; actions are already raw Gaussian samples."""
        return self.get_actions_log_prob(raw_actions)

    def evaluate(self, observations, privileged_observations, **kwargs):
        return self.critic(self._policy_input(observations, privileged_observations))

    def act_inference(self, observations, privileged_observations):
        return self.act_inference_raw(observations, privileged_observations)

    def act_inference_raw(self, observations, privileged_observations):
        policy_input = self._policy_input(observations, privileged_observations)
        return self.actor(policy_input)

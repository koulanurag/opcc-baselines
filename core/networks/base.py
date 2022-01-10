import torch
import torch.nn as nn


class Base:
    def __init__(self, env_name, dataset_name, obs_size, action_size,
                 deterministic, constant_prior, prior_scale):
        self.__env_name = env_name
        self.__dataset_name = dataset_name
        self.__obs_size = obs_size
        self.__action_size = action_size
        self.__deterministic = deterministic
        self.__constant_prior = constant_prior
        self.__prior_scale = prior_scale

        self.__obs_max = torch.ones(obs_size) * torch.inf
        self.__obs_max = nn.Parameter(self.__obs_max, requires_grad=False)

        self.__obs_min = torch.ones(obs_size) * -torch.inf
        self.__obs_min = nn.Parameter(self.__obs_min, requires_grad=False)

        self.__reward_max = torch.ones(1) * torch.inf
        self.__reward_max = nn.Parameter(self.__reward_max, requires_grad=False)

        self.__reward_min = torch.ones(1) * -torch.inf
        self.__reward_min = nn.Parameter(self.__reward_min, requires_grad=False)

    @property
    def env_name(self):
        return self.__env_name

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def obs_size(self):
        return self.__obs_size

    @property
    def action_size(self):
        return self.__action_size

    @property
    def deterministic(self):
        return self.__deterministic

    @property
    def constant_prior(self):
        return self.__constant_prior

    @property
    def prior_scale(self):
        return self.__prior_scale

    @property
    def obs_min(self):
        return self.__obs_min

    @property
    def obs_max(self):
        return self.__obs_max

    @property
    def reward_min(self):
        return self.__reward_min

    @property
    def reward_max(self):
        return self.__reward_max

    def set_obs_bound(self, obs_min, obs_max):
        self.__obs_min = nn.Parameter(obs_min, requires_grad=False)
        self.__obs_max = nn.Parameter(obs_max, requires_grad=False)

    def set_reward_bound(self, reward_min, reward_max):
        self.__reward_min = nn.Parameter(reward_min, requires_grad=False)
        self.__reward_max = nn.Parameter(reward_max, requires_grad=False)

    def clip_obs(self, obs):
        raise NotImplementedError

    def clip_reward(self, reward):
        raise NotImplementedError

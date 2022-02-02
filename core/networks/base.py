import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(self, env_name, dataset_name, obs_size, action_size,
                 deterministic, prior_scale):
        super().__init__()
        assert prior_scale >= 0, 'prior scale must be +ve'
        self._env_name = env_name
        self._dataset_name = dataset_name
        self._obs_size = obs_size
        self._action_size = action_size
        self._deterministic = deterministic
        self._prior_scale = prior_scale
        self._clip_obs = False
        self._clip_reward = False

    def forward(self, obs, action):
        raise NotImplementedError

    def step(self, obs, action):
        raise NotImplementedError

    @property
    def env_name(self):
        return self._env_name

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def obs_size(self):
        return self._obs_size

    @property
    def action_size(self):
        return self._action_size

    @property
    def deterministic(self):
        return self._deterministic

    @property
    def prior_scale(self):
        return self._prior_scale

    def enable_obs_clip(self):
        self._clip_obs = True

    def enable_reward_clip(self):
        self._clip_reward = True

    def disable_obs_clip(self):
        self._clip_obs = False

    def disable_reward_clip(self):
        self._clip_reward = False

    @property
    def is_obs_clip_enabled(self):
        return self._clip_obs

    @property
    def is_reward_clip_enabled(self):
        return self._clip_reward

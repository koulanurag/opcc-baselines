import torch
import torch.nn as nn
from torch.nn import functional as F

from . import weights_init


class FeedForwardObsDynamicsNetwork(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size, deterministic=True,
                 activation_function='relu'):
        super().__init__()
        # Todo: Make comparison with this
        # https://github.com/Xingyu-Lin/mbpo_pytorch/blob/43c8a55fa7353c6aed97525d0ecd5cb903b55377/model.py#L113
        self.obs_size = obs_size
        self.num_actions = action_size
        self.deterministic = deterministic
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(obs_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2 * obs_size + 2)
        self.terminal_logit = nn.Sequential(nn.Linear(obs_size, hidden_size),
                                            nn.ELU(),
                                            nn.Linear(hidden_size, 1))

        self.max_logvar = nn.Parameter((torch.ones((1, obs_size + 1)).float() / 2),
                                       requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, obs_size + 1)).float() * 10),
                                       requires_grad=False)
        self.apply(weights_init)

    def terminal(self, observation):
        logit = self.terminal_logit(observation)
        return torch.sigmoid(logit)

    def to(self, device, **kwargs):
        self.max_logvar = self.max_logvar.to(device)
        self.min_logvar = self.min_logvar.to(device)
        return super(FeedForwardObsDynamicsNetwork, self).to(device, **kwargs)

    def logits(self, obs, action):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2

        hidden = self.act_fn(self.fc1(torch.cat((obs, action), dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        output = self.fc4(hidden)
        mu, log_var_logit = output[:, :self.obs_size + 1], output[:, self.obs_size + 1:]
        return mu, log_var_logit

    def forward(self, obs, action):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2

        hidden = self.act_fn(self.fc1(torch.cat((obs, action), dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        output = self.fc4(hidden)
        mu, log_var_logit = output[:, :self.obs_size + 1], output[:, self.obs_size + 1:]
        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var_logit)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        next_obs_mu, reward_mu = mu[:, :-1], mu[:, -1]
        next_obs_log_var, reward_log_var = log_var[:, :-1], log_var[:, -1]

        if self.deterministic:
            next_obs = next_obs_mu
            reward = reward_mu
        else:
            var = torch.exp(next_obs_log_var)
            next_obs = torch.normal(next_obs_mu, torch.sqrt(var))

            var = torch.exp(reward_log_var)
            reward = torch.normal(reward_mu, torch.sqrt(var))

        return next_obs, next_obs_mu, next_obs_log_var, reward, reward_mu, reward_log_var

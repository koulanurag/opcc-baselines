import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import Base
import numpy as np


def weights_init(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape),
                                                        mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear:
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class FeedForwardDynamicsNetwork(Base, nn.Module):
    def __init__(self, obs_size, action_size, hidden_size, deterministic=True,
                 activation_function='relu', lr=1e-3):
        Base.__init__(self, obs_size=obs_size,
                      action_size=action_size,
                      deterministic=deterministic)
        nn.Module.__init__(self)

        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(self.obs_size + self.action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2 * obs_size + 2)

        max_logvar = (torch.ones((1, obs_size + 1)).float() / 2)
        min_logvar = (-torch.ones((1, obs_size + 1)).float() * 10)
        self.max_logvar = nn.Parameter(max_logvar, requires_grad=False)
        self.min_logvar = nn.Parameter(min_logvar, requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.apply(weights_init)

    def to(self, device, **kwargs):
        self.max_logvar = self.max_logvar.to(device)
        self.min_logvar = self.min_logvar.to(device)
        return super(FeedForwardDynamicsNetwork, self).to(device, **kwargs)

    def logits(self, obs, action):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2

        hidden = self.act_fn(self.fc1(torch.cat((obs, action), dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        output = self.fc4(hidden)

        mu = output[:, :self.obs_size + 1]
        log_var_logit = output[:, self.obs_size + 1:]
        return mu, log_var_logit

    def forward(self, obs, action):
        mu, log_var_logit = self.logits(obs, action)
        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var_logit)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        return mu, log_var

    def step(self, obs, action):
        mu, log_var = self.forward(obs, action)
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

        return next_obs, next_obs_mu, next_obs_log_var, reward, \
               reward_mu, reward_log_var

    def update(self, obs, action, next_obs, reward):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2
        assert len(reward.shape) == 1
        assert (len(obs) == len(action) == len(reward)), \
            'batch size is not same'

        obs = obs.contiguous()
        action = action.contiguous()
        next_obs = next_obs.contiguous()
        reward = reward.contiguous()
        target = torch.cat((next_obs, reward.unsqueeze(1)), dim=1)

        mu, log_var = self.forward(obs, action)
        assert len(mu.shape) == len(log_var.shape) == 2

        if not self.deterministic:
            inv_var = torch.exp(-log_var)
            mse_loss = torch.mean(torch.mean(torch.pow(mu - target, 2)
                                             * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(log_var, dim=-1), dim=-1)
            loss = mse_loss + var_loss
            loss += 0.01 * torch.sum(self.max_logvar)
            loss -= 0.01 * torch.sum(self.min_logvar)
        else:
            loss = nn.MSELoss()(mu, target)

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'total': loss.item()}

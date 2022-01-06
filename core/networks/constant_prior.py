import torch
import torch.nn as nn
from torch.nn import functional as F
from ff import FeedForwardDynamicsNetwork
from base import Base


class ConstantPriorFeedForwardDynamicsNetwork(Base, nn.Module):
    def __init__(self, obs_size, action_size, hidden_size, deterministic=True,
                 activation_function='relu'):
        super().__init__(obs_size, action_size, deterministic)

        self.constant_prior = FeedForwardDynamicsNetwork(obs_size,
                                                         action_size,
                                                         hidden_size,
                                                         deterministic,
                                                         activation_function)
        self.dynamics = FeedForwardDynamicsNetwork(obs_size, action_size,
                                                   hidden_size,
                                                   deterministic,
                                                   activation_function)
        for param in self.constant_prior.parameters():
            param.requires_grad = False

        self.max_logvar = nn.Parameter(
            (torch.ones((1, obs_size + 1)).float() / 2), requires_grad=False)
        self.min_logvar = nn.Parameter(
            (-torch.ones((1, obs_size + 1)).float() * 10), requires_grad=False)

    def to(self, device):
        self.max_logvar = self.max_logvar.to(device)
        self.min_logvar = self.min_logvar.to(device)
        return super(ConstantPriorFeedForwardDynamicsNetwork, self).to(device)

    def logits(self, obs, action):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2

        pred_mu, pred_log_var_logit = self.dynamics.logits(obs, action)
        with torch.no_grad():
            prior_mu, prior_log_var_logit = self.constant_prior.logits(obs,
                                                                       action)

        mu = pred_mu + prior_mu
        log_var_logit = pred_log_var_logit + prior_log_var_logit

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

import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import Base
import numpy as np
from .utils import weights_init


class FFDynamicsNetwork(Base, nn.Module):
    """
    Feed-forward dynamics network
    """

    def __init__(self, obs_size, action_size, hidden_size, deterministic=True,
                 constant_prior=False, activation_function='relu', lr=1e-3):
        Base.__init__(self, obs_size=obs_size,
                      action_size=action_size,
                      deterministic=deterministic,
                      constant_prior=constant_prior)
        nn.Module.__init__(self)
        self.__prior_prefix = 'prior_'
        prefixes = [] if constant_prior else [self.__prior_prefix]

        self.act_fn = getattr(F, activation_function)
        for prefix in [''] + prefixes:
            setattr(self, prefix + 'fc1',
                    nn.Linear(self.obs_size + self.action_size, hidden_size))
            setattr(self, prefix + 'fc2',
                    nn.Linear(hidden_size, hidden_size))
            setattr(self, prefix + 'fc3',
                    nn.Linear(hidden_size, hidden_size))
            setattr(self, prefix + 'fc4'.format(prefix),
                    nn.Linear(hidden_size, 2 * obs_size + 2))

        for name, param in self.named_parameters():
            if self.__prior_prefix in name:
                param.requires_grad = False

        max_logvar = (torch.ones((1, obs_size + 1)).float() / 2)
        min_logvar = (-torch.ones((1, obs_size + 1)).float() * 10)
        self.max_logvar = nn.Parameter(max_logvar, requires_grad=False)
        self.min_logvar = nn.Parameter(min_logvar, requires_grad=False)
        self.optimizer = torch.optim.Adam([param
                                           for name, param in
                                           self.named_parameters()
                                           if self.__prior_prefix not in name],
                                          lr=lr)
        self.apply(weights_init)

    def to(self, device, **kwargs):
        self.max_logvar = self.max_logvar.to(device)
        self.min_logvar = self.min_logvar.to(device)
        return super(FFDynamicsNetwork, self).to(device, **kwargs)

    def __prior_logits(self, obs, action):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2

        hidden = self.act_fn(self.prior_fc1(torch.cat((obs, action), dim=1)))
        hidden = self.act_fn(self.prior_fc2(hidden))
        hidden = self.act_fn(self.prior_fc3(hidden))
        output = self.prior_fc4(hidden)

        mu = output[:, :self.obs_size + 1]
        log_var_logit = output[:, self.obs_size + 1:]
        return mu, log_var_logit

    def __logits(self, obs, action):
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
        mu, log_var_logit = self.__logits(obs, action)
        if self.constant_prior:
            with torch.no_grad():
                prior_mu, prior_log_var_logit = self.__prior_logits(obs,
                                                                    action)
                mu += prior_mu
                log_var_logit += prior_log_var_logit

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

import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import Base
from .utils import weights_init, is_terminal


class AgDynamicsNetwork(Base, nn.Module):
    """
    Autoregressive Dynamics Network
    """

    def __init__(self, env_name, dataset_name, obs_size, action_size,
                 hidden_size, deterministic=True, constant_prior=False,
                 activation_function='relu', lr=1e-3, prior_scale=1):
        Base.__init__(self,
                      env_name=env_name,
                      dataset_name=dataset_name,
                      obs_size=obs_size,
                      action_size=action_size,
                      deterministic=deterministic,
                      constant_prior=constant_prior,
                      prior_scale=prior_scale)
        nn.Module.__init__(self)

        self.act_fn = getattr(F, activation_function)
        self.__input_size = 3 * self.obs_size + self.action_size + 1
        self.fc1 = nn.Linear(self.__input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)

        self._prior_prefix = 'prior_'
        if constant_prior:
            layer = nn.Linear(self.__input_size, hidden_size)
            setattr(self, self._prior_prefix + 'fc1', layer)
            layer = nn.Linear(hidden_size, hidden_size)
            setattr(self, self._prior_prefix + 'fc2', layer)
            layer = nn.Linear(hidden_size, hidden_size)
            setattr(self, self._prior_prefix + 'fc3', layer)
            layer = nn.Linear(hidden_size, 2)
            setattr(self, self._prior_prefix + 'fc4', layer)

            for name, param in self.named_parameters():
                if self._prior_prefix in name:
                    param.requires_grad = False

        max_logvar = (torch.ones((1, 1)).float() / 2)
        min_logvar = (-torch.ones((1, 1)).float() * 10)
        self.max_logvar = nn.Parameter(max_logvar, requires_grad=False)
        self.min_logvar = nn.Parameter(min_logvar, requires_grad=False)
        self.apply(weights_init)

        non_prior_params = [param for name, param in self.named_parameters()
                            if self._prior_prefix not in name]
        self.optimizer = torch.optim.Adam(non_prior_params, lr=lr)

    def to(self, device, *args, **kwargs):
        self.max_logvar.data = self.max_logvar.to(device)
        self.min_logvar.data = self.min_logvar.to(device)
        return super(AgDynamicsNetwork, self).to(device, *args, **kwargs)

    def __prior_logits(self, obs, action):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2

        hidden = self.act_fn(self.prior_fc1(torch.cat((obs, action), dim=1)))
        hidden = self.act_fn(self.prior_fc2(hidden))
        hidden = self.act_fn(self.prior_fc3(hidden))
        output = self.prior_fc4(hidden)

        mu = output[:, 0]
        log_var_logit = output[:, 1]
        return mu, log_var_logit

    def _logits(self, obs, action):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2

        hidden = self.act_fn(self.fc1(torch.cat((obs, action), dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        output = self.fc4(hidden)

        mu = output[:, 0]
        log_var_logit = output[:, 1]
        return mu, log_var_logit

    def forward(self, obs, action):
        batch_size = obs.shape[0]
        next_obs = torch.zeros((batch_size, self.obs_size + 1)).to(obs.device)
        mu, log_var = None, None
        for obs_i in range(self.obs_size + 1):  # add dimension for reward

            # create obs
            one_hot = torch.zeros_like(obs).to(obs.device)
            one_hot[:, obs_i] = 1.0
            _obs = torch.cat((obs, next_obs.detach(), one_hot), dim=1)

            # ith dimension prediction
            mu_i, log_var_logit_i = self._logits(_obs, action)
            if self.constant_prior:
                with torch.no_grad():
                    prior_mu, prior_log_var_logit = self.__prior_logits(_obs,
                                                                        action)
                    mu_i += prior_mu
                    log_var_logit_i += prior_log_var_logit
            log_var_i = self.max_logvar - F.softplus(self.max_logvar - log_var_logit_i)
            log_var_i = self.min_logvar + F.softplus(log_var_i - self.min_logvar)

            if obs_i == 0:
                mu = mu_i
                log_var = log_var_i
            else:
                mu = torch.cat((mu, mu_i), dim=1)
                log_var = torch.cat((log_var, log_var_i), dim=1)

            # sample next obs ith dimension
            if self.deterministic:
                next_obs[:, obs_i] = mu_i.detach()
            else:
                var = torch.exp(log_var_i).detach()
                next_obs[:, obs_i] = torch.normal(mu_i.detach(),
                                                  torch.sqrt(var))
        reward = next_obs[:, -1]
        next_obs = next_obs[:, :-1]
        return next_obs, reward, mu, log_var

    def step(self, obs, action):
        # Todo: normalize obs. and action
        next_obs, reward, _, _ = self.forward(obs, action)

        # Todo: denormalize obs
        return next_obs, reward, is_terminal(self.env_name,
                                             next_obs.cpu().detach())

    def update(self, obs, action, next_obs, reward):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2
        assert len(reward.shape) == 1
        assert (len(obs) == len(action) == len(reward)), \
            'batch size is not same'

        obs = obs.contiguous()
        action = action.contiguous()
        next_obs = next_obs.contiguous()
        delta_obs = next_obs.detach() - obs.detach()
        reward = reward.contiguous()
        target = torch.cat((delta_obs, reward.unsqueeze(1)), dim=1)

        _, _, mu, log_var = self.forward(obs, action)
        assert len(mu.shape) == len(log_var.shape) == 2

        if self.deterministic:
            loss = nn.MSELoss()(mu, target)
        else:
            inv_var = torch.exp(-log_var)
            mse_loss = torch.mean(torch.pow(mu - target, 2) * inv_var)
            var_loss = torch.mean(log_var)
            loss = mse_loss + var_loss
            loss += 0.01 * torch.sum(self.max_logvar)
            loss -= 0.01 * torch.sum(self.min_logvar)

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'total': loss.item()}

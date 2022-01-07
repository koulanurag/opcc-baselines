import torch
import torch.nn as nn
from torch.nn import functional as F

from .ff import FFDynamicsNetwork
from .utils import weights_init


class AgDynamicsNetwork(FFDynamicsNetwork):
    """
    Autoregressive Dynamics Network
    """

    def __init__(self, obs_size, action_size, hidden_size, deterministic=True,
                 constant_prior=False, activation_function='relu', lr=1e-3):
        # obs = obs + next_obs + action + one_hot for obs and reward
        super().__init__(3 * obs_size, action_size + 1, hidden_size,
                         deterministic=deterministic,
                         constant_prior=constant_prior,
                         activation_function=activation_function,
                         lr=lr)

        setattr(self, 'fc4', nn.Linear(hidden_size, 2))
        if self.constant_prior:
            setattr(self, self._prior_prefix + 'fc4', nn.Linear(hidden_size,
                                                                2))

        for name, param in self.named_parameters():
            if self._prior_prefix in name:
                param.requires_grad = False

        max_logvar = (torch.ones((1, 1).float() / 2))
        min_logvar = (-torch.ones((1, 1)).float() * 10)
        self.max_logvar = nn.Parameter(max_logvar, requires_grad=False)
        self.min_logvar = nn.Parameter(min_logvar, requires_grad=False)
        self.optimizer = torch.optim.Adam([param
                                           for name, param in
                                           self.named_parameters()
                                           if self._prior_prefix not in name],
                                          lr=lr)
        self.apply(weights_init)

    def forward(self, obs, action):
        next_obs = torch.zeros_like(obs).to(obs.device)
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
            log_var_i = self.max_logvar - F.softplus(self.max_logvar
                                                     - log_var_logit_i)
            log_var_i = self.min_logvar + F.softplus(log_var_i
                                                     - self.min_logvar)

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

        return mu, log_var

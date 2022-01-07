from .ff import FFDynamicsNetwork
import torch
import torch.nn as nn
from .utils import weights_init


class AgDynamicsNetwork(FFDynamicsNetwork):
    """
    Autoregressive Dynamics Network
    """

    def __init__(self, obs_size, action_size, hidden_size, deterministic=True,
                 constant_prior=False, activation_function='relu', lr=1e-3):
        super().__init__(2 * obs_size, action_size, hidden_size,
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

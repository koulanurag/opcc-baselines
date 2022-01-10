import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import Base
from .utils import weights_init, is_terminal


class FFDynamicsNetwork(Base, nn.Module):
    """
    Feed-forward dynamics network
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
        self.fc1 = nn.Linear(self.obs_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2 * obs_size + 2)

        self._prior_prefix = 'prior_'
        if constant_prior:
            layer = nn.Linear(self.obs_size + action_size, hidden_size)
            setattr(self, self._prior_prefix + 'fc1', layer)
            layer = nn.Linear(hidden_size, hidden_size)
            setattr(self, self._prior_prefix + 'fc2', layer)
            layer = nn.Linear(hidden_size, hidden_size)
            setattr(self, self._prior_prefix + 'fc3', layer)
            layer = nn.Linear(hidden_size, 2 * obs_size + 2)
            setattr(self, self._prior_prefix + 'fc4', layer)

            for name, param in self.named_parameters():
                if self._prior_prefix in name:
                    param.requires_grad = False

        max_logvar = (torch.ones((1, obs_size + 1)).float() / 2)
        min_logvar = (-torch.ones((1, obs_size + 1)).float() * 10)
        self.max_logvar = nn.Parameter(max_logvar, requires_grad=False)
        self.min_logvar = nn.Parameter(min_logvar, requires_grad=False)
        self.apply(weights_init)

        # create optimizer with no prior parameters
        non_prior_params = [param for name, param in self.named_parameters()
                            if self._prior_prefix not in name]
        self.optimizer = torch.optim.Adam(non_prior_params, lr=lr)

    def to(self, device, *args, **kwargs):
        self.max_logvar.data = self.max_logvar.to(device)
        self.min_logvar.data = self.min_logvar.to(device)
        return super(FFDynamicsNetwork, self).to(device, *args, **kwargs)

    def __prior_logits(self, obs, action):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2

        hidden = torch.cat((obs, action), dim=1)
        hidden = self.act_fn(getattr(self, self._prior_prefix + 'fc1')(hidden))
        hidden = self.act_fn(getattr(self, self._prior_prefix + 'fc2')(hidden))
        hidden = self.act_fn(getattr(self, self._prior_prefix + 'fc3')(hidden))
        output = getattr(self, self._prior_prefix + 'fc4')(hidden)

        mu = output[:, :self.obs_size + 1]
        log_var_logit = output[:, self.obs_size + 1:]
        return mu, log_var_logit

    def _logits(self, obs, action):
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
        mu, log_var_logit = self._logits(obs, action)
        if self.constant_prior:
            with torch.no_grad():
                _mu, _log_var_logit = self.__prior_logits(obs, action)
                mu += self.prior_scale * _mu
                log_var_logit += self.prior_scale * _log_var_logit
        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var_logit)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        return mu, log_var

    def step(self, obs, action):
        # Todo: normalize obs. and action
        mu, log_var = self.forward(obs, action)
        delta_next_obs_mu, reward_mu = mu[:, :-1], mu[:, -1]
        next_obs_log_var, reward_log_var = log_var[:, :-1], log_var[:, -1]

        if self.deterministic:
            delta_next_obs = delta_next_obs_mu
            reward = reward_mu
        else:
            var = torch.exp(next_obs_log_var)
            delta_next_obs = torch.normal(delta_next_obs_mu, torch.sqrt(var))

            var = torch.exp(reward_log_var)
            reward = torch.normal(reward_mu, torch.sqrt(var))

        next_obs = obs.detach() + delta_next_obs

        # Todo: denormalize obs
        next_obs = self.clip_obs(next_obs)
        reward = self.clip_reward(reward)
        done = is_terminal(self.env_name, next_obs.cpu().detach())
        return next_obs, reward, done

    def update(self, obs, action, next_obs, reward):
        assert len(obs.shape) == 2
        assert len(action.shape) == 2
        assert len(next_obs.shape) == 2
        assert len(reward.shape) == 2
        assert ((len(obs) == len(action) == len(next_obs) == len(reward)),
                'batch size is not same')

        obs = obs.contiguous()
        action = action.contiguous()
        next_obs = next_obs.contiguous()
        delta_obs = next_obs.detach() - obs.detach()
        reward = reward.contiguous()
        target = torch.cat((delta_obs, reward), dim=1)

        mu, log_var = self.forward(obs, action)
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

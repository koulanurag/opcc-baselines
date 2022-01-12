import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import Base
from .utils import weights_init, is_terminal


class AgDynamicsNetwork(Base):
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

        self.act_fn = getattr(F, activation_function)
        self._input_size = 3 * self.obs_size + self.action_size + 1
        self.fc1 = nn.Linear(self._input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)

        if constant_prior:
            layer = nn.Linear(self.obs_size + action_size, hidden_size)
            self.prior_fc1 = layer
            self.prior_fc2 = nn.Linear(hidden_size, hidden_size)
            self.prior_fc3 = nn.Linear(hidden_size, hidden_size)
            self.prior_fc4 = nn.Linear(hidden_size, 2)
            for name, param in self.named_parameters():
                if 'prior' in name:
                    param.requires_grad = False

        self.apply(weights_init)

        max_logvar = torch.ones(1, dtype=torch.float) / 2
        min_logvar = -torch.ones(1, dtype=torch.float) * 10
        self.max_logvar = nn.Parameter(max_logvar, requires_grad=False)
        self.min_logvar = nn.Parameter(min_logvar, requires_grad=False)

        # default bounds
        self._obs_max = torch.ones(obs_size, dtype=torch.float) * torch.inf
        self._obs_max = nn.Parameter(self._obs_max, requires_grad=False)
        self._obs_min = torch.ones(obs_size, dtype=torch.float) * -torch.inf
        self._obs_min = nn.Parameter(self._obs_min, requires_grad=False)

        self._reward_max = torch.ones(1, dtype=torch.float) * torch.inf
        self._reward_max = nn.Parameter(self._reward_max, requires_grad=False)
        self._reward_min = torch.ones(1, dtype=torch.float) * -torch.inf
        self._reward_min = nn.Parameter(self._reward_min, requires_grad=False)

        # create optimizer with no prior parameters
        non_prior_params = [param for name, param in self.named_parameters()
                            if 'prior' not in name]
        self.optimizer = torch.optim.Adam(non_prior_params, lr=lr)

    def to(self, device, *args, **kwargs):
        self.max_logvar.data = self.max_logvar.to(device)
        self.min_logvar.data = self.min_logvar.to(device)

        self._obs_max.data = self._obs_max.to(device)
        self._obs_min.data = self._obs_min.to(device)
        self._reward_max.data = self._reward_max.to(device)
        self._reward_min.data = self._reward_min.to(device)

        return super(AgDynamicsNetwork, self).to(device, *args, **kwargs)

    def _prior_logits(self, obs, action):
        assert len(obs.shape) == 2, 'expected (N x obs-size) observation'
        assert len(action.shape) == 2, 'expected (N x action-size) actions'

        hidden = torch.cat((obs, action), dim=1)
        hidden = self.act_fn(self.prior_fc1(hidden))
        hidden = self.act_fn(self.prior_fc2(hidden))
        hidden = self.act_fn(self.prior_fc3(hidden))
        output = self.prior_fc4(hidden)

        mu = output[:, 0]
        log_var_logit = output[:, 1]
        return mu, log_var_logit

    def _logits(self, obs, action):
        assert len(obs.shape) == 2, 'expected (N x obs-size) observation'
        assert len(action.shape) == 2, 'expected (N x action-size) actions'

        hidden = self.act_fn(self.fc1(torch.cat((obs, action), dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        output = self.fc4(hidden)

        mu = output[:, 0]
        log_var_logit = output[:, 1]
        return mu, log_var_logit

    def forward(self, obs, action):
        batch_size = obs.shape[0]
        next_obs = torch.zeros((batch_size, self.obs_size)).to(obs.device)
        reward, mu, log_var = None, None, None
        for obs_i in range(self.obs_size + 1):  # add dimension for reward

            # create obs
            one_hot = torch.zeros((batch_size, self.obs_size + 1)).to(obs.device)
            one_hot[:, obs_i] = 1.0
            _obs = torch.cat((obs, next_obs.detach(), one_hot), dim=1)

            # ith dimension prediction
            mu_i, log_var_logit_i = self._logits(_obs, action)
            if self.constant_prior:
                with torch.no_grad():
                    _mu_i, _log_var_logit_i = self._prior_logits(_obs, action)
                    mu_i += self.prior_scale * _mu_i
                    log_var_logit_i += self.prior_scale * _log_var_logit_i
            log_var_i = self.max_logvar - F.softplus(self.max_logvar - log_var_logit_i)
            log_var_i = self.min_logvar + F.softplus(log_var_i - self.min_logvar)

            if obs_i == 0:
                mu = mu_i.unsqueeze(-1)
                log_var = log_var_i.unsqueeze(-1)
            else:
                mu = torch.cat((mu, mu_i.unsqueeze(-1)), dim=1)
                log_var = torch.cat((log_var, log_var_i.unsqueeze(-1)), dim=1)

            # sample next obs ith dimension
            if self.deterministic:
                output = mu_i.detach()
            else:
                var = torch.exp(log_var_i).detach()
                output = torch.normal(mu_i.detach(), torch.sqrt(var))

            # clip
            if obs_i < self.obs_size:
                # output is the delta observation
                next_obs[:, obs_i] = output + obs[:, obs_i]
                if self.is_obs_clip_enabled:
                    next_obs[:, obs_i] = self.clip_obs(next_obs[:, obs_i], obs_i)
            else:
                if self.is_reward_clip_enabled:
                    reward = self.clip_reward(output)
                else:
                    reward = output

        return next_obs, reward, mu, log_var

    def step(self, obs, action):
        # Todo: normalize obs. and action
        next_obs, reward, mu, log_var = self.forward(obs, action)
        # Todo: de-normalize obs. and action
        done = is_terminal(self.env_name, next_obs.cpu().detach())
        return next_obs, reward, done

    def update(self, obs, action, next_obs, reward):
        assert len(obs.shape) == 2, 'expected (N x obs-size) observation'
        assert len(action.shape) == 2, 'expected (N x action-size) actions'
        assert len(next_obs.shape) == 2, 'expected (N x obs-size) observation'
        assert len(reward.shape) == 2, 'expected (N x 1) reward'
        assert len(obs) == len(action) == len(next_obs) == len(reward), \
            'batch size is not same'

        obs = obs.contiguous()
        action = action.contiguous()
        next_obs = next_obs.contiguous()
        reward = reward.contiguous()

        delta_obs = next_obs.detach() - obs.detach()
        target = torch.cat((delta_obs, reward), dim=1)

        _, _, mu, log_var = self.forward(obs, action)
        assert len(mu.shape) == len(log_var.shape) == 2

        if self.deterministic:
            loss = torch.mean(torch.pow(mu - target, 2))
        else:
            inv_var = torch.exp(-log_var)
            mse_loss = torch.mean(torch.pow(mu - target, 2) * inv_var)
            var_loss = torch.mean(log_var)
            var_loss += 0.01 * torch.sum(self.max_logvar)
            var_loss -= 0.01 * torch.sum(self.min_logvar)
            loss = mse_loss + var_loss

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'total': loss.item()}

    @property
    def obs_min(self):
        return self._obs_min

    @property
    def obs_max(self):
        return self._obs_max

    @property
    def reward_min(self):
        return self._reward_min

    @property
    def reward_max(self):
        return self._reward_max

    def set_obs_bound(self, obs_min, obs_max):
        obs_min = torch.tensor(obs_min)
        obs_max = torch.tensor(obs_max)
        self._obs_min = nn.Parameter(obs_min, requires_grad=False)
        self._obs_max = nn.Parameter(obs_max, requires_grad=False)

    def set_reward_bound(self, reward_min, reward_max):
        reward_min = torch.tensor(reward_min)
        reward_max = torch.tensor(reward_max)
        self._reward_min = nn.Parameter(reward_min, requires_grad=False)
        self._reward_max = nn.Parameter(reward_max, requires_grad=False)

    def clip_obs(self, obs, dim: int):
        assert 0 <= dim < self.obs_size
        return torch.clip(obs, min=self.obs_min[dim], max=self.obs_max[dim])

    def clip_reward(self, reward):
        return torch.clip(reward, min=self.reward_min, max=self.reward_max)

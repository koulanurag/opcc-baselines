import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import Base
from .utils import weights_init, is_terminal


class FFDynamicsNetwork(Base):
    """
    Feed-forward dynamics network

    Given an observation and action, the network predicts delta observation
    and reward. Thereafter, delta observation is added to the input observation
    to retrieve next observation.

    """

    def __init__(self, env_name, dataset_name, obs_size, action_size,
                 hidden_size, deterministic=True, activation_function='relu',
                 lr=1e-3, prior_scale=0):
        Base.__init__(self,
                      env_name=env_name,
                      dataset_name=dataset_name,
                      obs_size=obs_size,
                      action_size=action_size,
                      deterministic=deterministic,
                      prior_scale=prior_scale)

        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(self.obs_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2 * obs_size + 2)

        self.prior_fc1 = nn.Linear(self.obs_size + action_size, hidden_size)
        self.prior_fc2 = nn.Linear(hidden_size, hidden_size)
        self.prior_fc3 = nn.Linear(hidden_size, hidden_size)
        self.prior_fc4 = nn.Linear(hidden_size, 2 * obs_size + 2)

        for name, param in self.named_parameters():
            if 'prior' in name:
                param.requires_grad = False

        self.apply(weights_init)

        max_logvar = torch.ones((1, obs_size + 1), dtype=torch.float) / 2
        min_logvar = -torch.ones((1, obs_size + 1), dtype=torch.float) * 5
        self.max_logvar = nn.Parameter(max_logvar, requires_grad=False)
        self.min_logvar = nn.Parameter(min_logvar, requires_grad=False)

        # default bounds for clipping
        self._obs_max = torch.ones(obs_size, dtype=torch.float) * torch.inf
        self._obs_max = nn.Parameter(self._obs_max, requires_grad=False)
        self._obs_min = torch.ones(obs_size, dtype=torch.float) * -torch.inf
        self._obs_min = nn.Parameter(self._obs_min, requires_grad=False)

        self._reward_max = torch.tensor(torch.inf, dtype=torch.float)
        self._reward_max = nn.Parameter(self._reward_max, requires_grad=False)
        self._reward_min = torch.tensor(-torch.inf, dtype=torch.float)
        self._reward_min = nn.Parameter(self._reward_min, requires_grad=False)

        # default normalization attributes: only used during step()
        _obs_mean = torch.zeros(obs_size, dtype=torch.float)
        _obs_std = torch.ones(obs_size, dtype=torch.float)
        self._obs_mean = nn.Parameter(_obs_mean, requires_grad=False)
        self._obs_std = nn.Parameter(_obs_std, requires_grad=False)

        _action_mean = torch.zeros(action_size, dtype=torch.float)
        _action_std = torch.ones(action_size, dtype=torch.float)
        self._action_mean = nn.Parameter(_action_mean, requires_grad=False)
        self._action_std = nn.Parameter(_action_std, requires_grad=False)

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

        return super(FFDynamicsNetwork, self).to(device, *args, **kwargs)

    @torch.no_grad()
    def _prior_logits(self, obs, action):
        assert len(obs.shape) == 2, 'expected (N x obs-size) observation'
        assert len(action.shape) == 2, 'expected (N x action-size) actions'

        hidden = torch.cat((obs, action), dim=1)
        hidden = self.act_fn(self.prior_fc1(hidden))
        hidden = self.act_fn(self.prior_fc2(hidden))
        hidden = self.act_fn(self.prior_fc3(hidden))
        output = self.prior_fc4(hidden)
        output = torch.tanh(output)

        mu = output[:, :self.obs_size + 1]
        log_var_logit = output[:, self.obs_size + 1:]
        return mu, log_var_logit

    def _logits(self, obs, action):
        assert len(obs.shape) == 2, 'expected (N x obs-size) observation'
        assert len(action.shape) == 2, 'expected (N x action-size) actions'

        hidden = self.act_fn(self.fc1(torch.cat((obs, action), dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        output = self.fc4(hidden)

        mu = output[:, :self.obs_size + 1]
        log_var_logit = output[:, self.obs_size + 1:]
        return mu, log_var_logit

    def forward(self, obs, action):
        mu, log_var_logit = self._logits(obs, action)
        if self.prior_scale > 0:
            _mu, _log_var_logit = self._prior_logits(obs, action)
            mu += self.prior_scale * _mu.detach()
            log_var_logit += self.prior_scale * _log_var_logit.detach()
        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var_logit)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        return mu, log_var

    def step(self, obs, action):

        # normalize obs. and action
        obs = self.normalize_obs(obs)
        action = self.normalize_action(action)

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
        if self.is_obs_clip_enabled:
            next_obs = self.clip_obs(next_obs)
        if self.is_reward_clip_enabled:
            reward = self.clip_reward(reward)

        # denormalize next-obs
        # This must be done before determining terminal state
        next_obs = self.denormalize_obs(next_obs)
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

        mu, log_var = self.forward(obs, action)
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

    def set_obs_norm(self, obs_mean, obs_std):
        assert len(obs_mean) == self.obs_size
        assert len(obs_std) == self.obs_size

        obs_mean = torch.tensor(obs_mean)
        obs_std = torch.tensor(obs_std)

        self._obs_mean = nn.Parameter(obs_mean, requires_grad=False)
        self._obs_std = nn.Parameter(obs_std, requires_grad=False)

    def set_action_norm(self, action_mean, action_std):
        assert len(action_mean) == self.action_size
        assert len(action_std) == self.action_size

        action_mean = torch.tensor(action_mean)
        action_std = torch.tensor(action_std)

        self._action_mean = nn.Parameter(action_mean, requires_grad=False)
        self._action_std = nn.Parameter(action_std, requires_grad=False)

    def clip_obs(self, obs):
        return torch.clip(obs, min=self.obs_min, max=self.obs_max)

    def clip_reward(self, reward):
        return torch.clip(reward, min=self.reward_min, max=self.reward_max)

    def normalize_obs(self, obs):
        return self.normalize(obs, self._obs_mean, self._obs_std)

    def denormalize_obs(self, obs):
        return self.denormalize(obs, self._obs_mean, self._obs_std)

    def normalize_action(self, action):
        return self.normalize(action, self._action_mean, self._action_std)

    @staticmethod
    def normalize(x, mean, std):
        x_norm = (x - mean) / std
        return x_norm

    @staticmethod
    def denormalize(x_norm, mean, std):
        return (x_norm * std) + mean

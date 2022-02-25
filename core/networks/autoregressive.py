import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import Base
from .utils import weights_init, is_terminal


class AgDynamicsNetwork(Base):
    """
    Autoregressive dynamics network

    Given a d-dimensional observation and action(n), the network predicts
    delta observation and reward. Thereafter, delta observation is added
    to the input observation to retrieve next observation.

    This network makes "d" amount of forward passes to predict the next
    observation. In each pass, it predicts the ith dimension of next
    observation by inputting the current observation, action, one
    hot-representation encoding to indicate the "d" dimension to be predicted
    and the next-observation predicted so far i.e. it receives (3 * d + n)
    size input.

    Reference : https://arxiv.org/abs/2104.13877
    """

    def __init__(self, env_name, dataset_name, obs_size, action_size,
                 hidden_size, deterministic=True,
                 activation_function='relu', lr=1e-3, prior_scale=0):
        Base.__init__(self,
                      env_name=env_name,
                      dataset_name=dataset_name,
                      obs_size=obs_size,
                      action_size=action_size,
                      deterministic=deterministic,
                      prior_scale=prior_scale)

        self.act_fn = getattr(F, activation_function)
        self._input_size = 3 * self.obs_size + self.action_size + 1
        self.fc1 = nn.Linear(self._input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)

        self.prior_fc1 = nn.Linear(self.obs_size + action_size, hidden_size)
        self.prior_fc2 = nn.Linear(hidden_size, hidden_size)
        self.prior_fc3 = nn.Linear(hidden_size, hidden_size)
        self.prior_fc4 = nn.Linear(hidden_size, 2 * obs_size + 2)

        for name, param in self.named_parameters():
            if 'prior' in name:
                param.requires_grad = False

        self.apply(weights_init)

        max_logvar = torch.ones(1, dtype=torch.float) / 2
        min_logvar = -torch.ones(1, dtype=torch.float) * 5
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

        return super(AgDynamicsNetwork, self).to(device, *args, **kwargs)

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

        mu = output[:, 0]
        log_var_logit = output[:, 1]
        return mu, log_var_logit

    def forward(self, obs, action):
        batch_size = obs.shape[0]
        next_obs = torch.zeros((batch_size, self.obs_size),
                               dtype=obs.dtype, device=obs.device)
        one_hot = torch.zeros((batch_size, self.obs_size + 1),
                              dtype=obs.dtype, device=obs.device)  # create obs
        one_hot[:, 0] = 1.0

        # estimate prior
        prior_mu, prior_log_var_logit = self._prior_logits(obs, action)
        prior_mu = prior_mu.detach()
        prior_log_var_logit = prior_log_var_logit.detach()

        # predictions
        reward, mu, log_var = None, None, None
        for obs_i in range(self.obs_size + 1):  # add dimension for reward

            # create obs
            _obs = torch.cat((obs, next_obs.detach(), one_hot), dim=1)

            # ith dimension prediction
            mu_i, log_var_logit_i = self._logits(_obs, action)
            mu_i += self.prior_scale * prior_mu[:, obs_i]
            log_var_logit_i += self.prior_scale * prior_log_var_logit[:, obs_i]

            log_var_i = (self.max_logvar
                         - F.softplus(self.max_logvar - log_var_logit_i))
            log_var_i = (self.min_logvar
                         + F.softplus(log_var_i - self.min_logvar))

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
                # prediction are delta observation
                next_obs[:, obs_i] = output + obs[:, obs_i]
                if self.is_obs_clip_enabled:
                    next_obs[:, obs_i] = self.clip_obs(next_obs[:, obs_i],
                                                       obs_i)
            else:
                if self.is_reward_clip_enabled:
                    reward = self.clip_reward(output)
                else:
                    reward = output

            # update one-hot
            one_hot[:, obs_i] = 0.0
            if obs_i < self.obs_size:
                one_hot[:, obs_i + 1] = 1.0

        return next_obs, reward, mu, log_var

    def step(self, obs, action):
        # normalize obs. and action
        obs = self.normalize_obs(obs)
        action = self.normalize_action(action)

        next_obs, reward, mu, log_var = self.forward(obs, action)

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

    def clip_obs(self, obs, dim: int):
        assert 0 <= dim < self.obs_size
        return torch.clip(obs, min=self.obs_min[dim], max=self.obs_max[dim])

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

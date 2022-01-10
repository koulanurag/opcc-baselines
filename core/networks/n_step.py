from collections import defaultdict

import torch

from .autoregressive import AgDynamicsNetwork
from .ff import FFDynamicsNetwork


class NstepDynamicsNetwork:
    def __init__(self, env_name, dataset_name, obs_size, action_size, hidden_size, n_step=1,
                 dynamics_type='feed-forward', deterministic=True,
                 constant_prior=False, prior_scale=1.0):
        super().__init__()
        self.__n_step = n_step
        for i in range(n_step):
            if dynamics_type == 'feed-forward':
                net = FFDynamicsNetwork(env_name, dataset_name,
                                        obs_size,
                                        (action_size * (i + 1)),
                                        hidden_size=hidden_size,
                                        constant_prior=constant_prior,
                                        deterministic=deterministic,
                                        activation_function='silu',
                                        prior_scale=prior_scale)

            elif dynamics_type == 'autoregressive':
                net = AgDynamicsNetwork(env_name, dataset_name,
                                        obs_size,
                                        (action_size * (i + 1)),
                                        hidden_size=hidden_size,
                                        constant_prior=constant_prior,
                                        deterministic=deterministic,
                                        activation_function='silu',
                                        prior_scale=prior_scale)
            else:
                raise ValueError()
            setattr(self, 'step_{}'.format(i + 1), net)

        self._max_steps = None
        self._action_history = None
        self._init_obs = None
        self._step_count = None
        self._batch_size = None
        self._dones = None

    def reset(self, max_steps=1, batch_size=1):
        self._max_steps = max_steps
        self._batch_size = batch_size
        self._dones = torch.Tensor([False for _ in range(batch_size)]).bool()
        self._step_count = 0
        self._action_history = None
        self._init_obs = None

    def step(self, obs, action):
        if self._max_steps is None:
            raise Exception('need to call reset() before step()')
        assert obs.shape[0] == action.shape[0] == self._batch_size
        if self._step_count % self.n_step == 0:
            self._init_obs = obs
            self._action_history = action
        else:
            self._action_history = torch.cat((self._action_history, action),
                                             dim=1)

        step_i = (self._step_count % self.n_step) + 1
        dynamics = getattr(self, 'step_{}'.format(step_i))
        next_obs, reward, done = dynamics.step(self._init_obs,
                                               self._action_history)
        self._step_count += 1
        if ((self._step_count % self.n_step) != 0 and
                self._step_count != self._max_steps):
            reward = torch.zeros_like(reward)
        else:
            reward[self._dones] = 0.

        self._dones = torch.logical_or(self._dones, done)
        return next_obs, reward, self._dones

    def update(self, replay_buffer, batch_count, batch_size):
        loss = defaultdict(lambda: defaultdict(lambda: 0))
        for batch_i in range(batch_count):
            batch = replay_buffer.sample(batch_size, self.n_step)

            init_obs = batch.obs[:, 0]
            dones = torch.zeros(batch_size).bool()
            for i in range(self.n_step):
                _name = 'step_{}'.format(i + 1)
                dynamics = getattr(self, _name)

                act = batch.action[:, :i + 1][dones]
                act = act.view(act.shape[0], act.shape[1] * act.shape[2])
                next_obs = batch.obs[:, i + 1][dones]
                reward = batch.reward[:, :i + 1][dones].sum(dim=1).unsqueeze(-1)

                _loss = dynamics.update(init_obs[dones], act, next_obs, reward)
                for k, v in _loss.items():
                    loss[_name][k] += v

                dones = torch.logical_or((dones, batch.terminal[:, i],
                                          batch.timeout[:, i]))

        # mean with batch count
        for k in loss:
            for _k, _v in loss[k].items():
                loss[k][_k] = _v / batch_count

        return loss

    @property
    def n_step(self):
        return self.__n_step

    def to(self, device, *args, **kwargs):
        for i in range(self.n_step):
            step_i = 'step_{}'.format(i + 1)
            setattr(self, step_i,
                    getattr(self, step_i).to(device, *args, **kwargs))
        return self

    def train(self, *args, **kwargs):
        for i in range(self.n_step):
            getattr(self, 'step_{}'.format(i + 1)).train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        for i in range(self.n_step):
            getattr(self, 'step_{}'.format(i + 1)).eval(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        _dict = {}
        for i in range(self.n_step):
            name = 'step_{}'.format(i + 1)
            _dict[name]['network'] = getattr(self, name).state_dict(*args, **kwargs)
            _dict[name]['optimizer'] = getattr(self, name).optimizer.state_dict(*args, **kwargs)
        return _dict

    def load_state_dict(self, state_dict):
        for i in range(self.n_step):
            name = 'step_{}'.format(i + 1)
            getattr(self, name).load_state_dict(state_dict[name]['network'])
            getattr(self, name).optimizer.load_state_dict(state_dict[name]['optimizer'])

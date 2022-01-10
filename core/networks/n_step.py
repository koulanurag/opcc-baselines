from collections import defaultdict

import torch

from .autoregressive import AgDynamicsNetwork
from .ff import FFDynamicsNetwork


class NstepDynamicsNetwork:
    def __init__(self, env_name, dataset_name, obs_size, action_size,
                 hidden_size, n_step=1, dynamics_type='feed-forward',
                 deterministic=True, constant_prior=False, prior_scale=1.0):
        super().__init__()
        self.__n_step = n_step
        for i in range(n_step):
            if dynamics_type == 'feed-forward':
                net = FFDynamicsNetwork(env_name=env_name,
                                        dataset_name=dataset_name,
                                        obs_size=obs_size,
                                        action_size=(action_size * (i + 1)),
                                        hidden_size=hidden_size,
                                        deterministic=deterministic,
                                        activation_function='silu',
                                        constant_prior=constant_prior,
                                        prior_scale=prior_scale)

            elif dynamics_type == 'autoregressive':
                net = AgDynamicsNetwork(env_name=env_name,
                                        dataset_name=dataset_name,
                                        obs_size=obs_size,
                                        action_size=(action_size * (i + 1)),
                                        hidden_size=hidden_size,
                                        deterministic=deterministic,
                                        activation_function='silu',
                                        constant_prior=constant_prior,
                                        prior_scale=prior_scale)
            else:
                raise ValueError()
            setattr(self, 'step_{}'.format(i + 1), net)

        self.__horizon = None
        self._action_hist = None  # maintains n-step actions
        self._init_obs = None  # maintains initial observation
        self._step_count = None
        self._batch_size = None
        self._dones = None

    def reset(self, horizon=1, batch_size=1):
        self.__horizon = horizon
        self._batch_size = batch_size
        self._dones = torch.zeros(batch_size).bool()
        self._step_count = 0
        self._action_hist = None
        self._init_obs = None

    def step(self, obs, action):
        if self.__horizon is None:
            raise Exception('need to call reset() before step()')
        assert obs.shape[0] == action.shape[0] == self._batch_size

        # reset observation after every n-steps
        if self._step_count % self.n_step == 0:
            self._init_obs = obs
            self._action_hist = action
        else:
            self._action_hist = torch.cat((self._action_hist, action), dim=1)

        # step
        step_i = (self._step_count % self.n_step) + 1
        dynamics = getattr(self, 'step_{}'.format(step_i))
        next_obs, reward, done = dynamics.step(self._init_obs, self._action_hist)
        self._step_count += 1

        # return reward only on end of horizon or on meeting n-step dynamics
        if ((self._step_count % self.n_step) != 0 and
                self._step_count != self.__horizon):
            reward = torch.zeros_like(reward)
        else:
            reward[self._dones] = 0.  # if terminal has passed, reward is 0.

        # update terminal information
        self._dones = torch.logical_or(self._dones, done)

        return next_obs, reward, self._dones

    def update(self, replay_buffer, batch_count, batch_size):
        loss = defaultdict(lambda: defaultdict(lambda: 0))
        for batch_i in range(batch_count):
            batch = replay_buffer.sample(batch_size, self.n_step)
            dones = torch.zeros(batch_size).bool()

            # init-obs is passed to all n-step dynamics
            init_obs = batch.obs[:, 0]
            for i in range(self.n_step):
                _name = 'step_{}'.format(i + 1)
                dynamics = getattr(self, _name)
                obs = init_obs[~dones]

                if len(obs) == 0:
                    break
                else:
                    act = batch.action[:, :i + 1][~dones]
                    act = act.view(act.shape[0], act.shape[1] * act.shape[2])
                    next_obs = batch.obs[:, i + 1][~dones]

                    # n-step return
                    reward = batch.reward[:, :i + 1][~dones]
                    reward_sum = reward.sum(dim=1).unsqueeze(-1)

                    # update
                    _loss = dynamics.update(obs, act, next_obs, reward_sum)
                    for k, v in _loss.items():
                        loss[_name][k] += v

                    # once done; don't update subsequent n-step dynamics
                    _dones = torch.logical_or(dones, batch.terminal[:, i]),
                    dones = torch.logical_or(_dones, batch.timeout[:, i])

        # mean with batch count
        for k in loss:
            for _k, _v in loss[k].items():
                loss[k][_k] = _v / batch_count

        return loss

    def to(self, device, *args, **kwargs):
        for i in range(self.n_step):
            step_i = 'step_{}'.format(i + 1)
            dynamics = getattr(self, step_i).to(device, *args, **kwargs)
            setattr(self, step_i, dynamics)

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
            dynamics = getattr(self, name)
            _dict[name]['network'] = dynamics.state_dict(*args, **kwargs)
            _dict[name]['optimizer'] = dynamics.optimizer.state_dict(*args,
                                                                     **kwargs)
        return _dict

    def load_state_dict(self, state_dict):
        for i in range(self.n_step):
            name = 'step_{}'.format(i + 1)
            dynamics = getattr(self, name)
            dynamics.load_state_dict(state_dict[name]['network'])
            dynamics.optimizer.load_state_dict(state_dict[name]['optimizer'])

    @property
    def n_step(self):
        return self.__n_step

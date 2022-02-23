from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from gym.utils import seeding

from .autoregressive import AgDynamicsNetwork
from .ff import FFDynamicsNetwork


class EnsembleDynamicsNetwork:
    def __init__(self, env_name: str, dataset_name: str, num_ensemble,
                 obs_size, action_size, hidden_size, dynamics_type,
                 deterministic=True, prior_scale: float = 0):
        super().__init__()
        self._np_random = None
        self.__obs_size = obs_size
        self.__action_size = action_size
        self.__num_ensemble = num_ensemble
        self.__deterministic = deterministic
        self._ensemble_mixture = False
        for i in range(num_ensemble):
            if dynamics_type == 'feed-forward':
                net = FFDynamicsNetwork(env_name=env_name,
                                        dataset_name=dataset_name,
                                        obs_size=obs_size,
                                        action_size=action_size,
                                        hidden_size=hidden_size,
                                        deterministic=deterministic,
                                        activation_function='silu',
                                        prior_scale=prior_scale)

            elif dynamics_type == 'autoregressive':
                net = AgDynamicsNetwork(env_name=env_name,
                                        dataset_name=dataset_name,
                                        obs_size=obs_size,
                                        action_size=action_size,
                                        hidden_size=hidden_size,
                                        deterministic=deterministic,
                                        activation_function='silu',
                                        prior_scale=prior_scale)
            else:
                raise ValueError()
            setattr(self, 'ensemble_{}'.format(i), net)

    def enable_mixture(self, seed=None):
        self._ensemble_mixture = True
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def disable_mixture(self):
        self._ensemble_mixture = False

    def step(self, obs, action):
        assert len(obs.shape) == 3, '(batch , ensemble , obs. size) required.'
        assert len(action.shape) == 3, '(batch , ensemble , obs. ) required.'

        next_obs, reward, done = None, None, None

        if self._ensemble_mixture:
            # Todo: with replacement eventually
            ensemble_idxs = np.arange(0, self.num_ensemble)
            self._np_random.shuffle(ensemble_idxs)
        else:
            ensemble_idxs = [_ for _ in range(self.num_ensemble)]

        for i in range(self.num_ensemble):
            _name = 'ensemble_{}'.format(ensemble_idxs[i])
            dynamics = getattr(self, _name)

            _next_obs, _reward, _done = dynamics.step(obs[:, i], action[:, i])
            _next_obs = _next_obs.unsqueeze(1)
            _reward = _reward.unsqueeze(1)
            _done = _done.unsqueeze(1)

            if i == 0:
                next_obs, reward, done = _next_obs, _reward, _done
            else:
                next_obs = torch.cat((next_obs, _next_obs), dim=1)
                reward = torch.cat((reward, _reward), dim=1)
                done = torch.cat((done, _done), dim=1)

        return next_obs, reward, done

    def update(self, replay_buffers, update_count: int,
               batch_size: int) -> Dict[str, Dict[str, float]]:
        loss = defaultdict(lambda: defaultdict(lambda: 0))
        for i in range(self.num_ensemble):
            _name = 'ensemble_{}'.format(i)
            dynamics = getattr(self, _name)

            for batch_i in range(update_count):
                batch = replay_buffers[i].sample(batch_size)
                _loss = dynamics.update(batch.obs,
                                        batch.action,
                                        batch.next_obs,
                                        batch.reward.unsqueeze(-1))
                for k, v in _loss.items():
                    loss[_name][k] += v

        # mean with update count
        for ensemble_key in loss:
            for loss_key in loss[ensemble_key]:
                loss[ensemble_key][loss_key] /= update_count
        return loss

    @property
    def num_ensemble(self):
        return self.__num_ensemble

    @property
    def deterministic(self):
        return self.__deterministic

    def to(self, device, *args, **kwargs):
        for i in range(self.num_ensemble):
            ensemble_i = 'ensemble_{}'.format(i)
            setattr(self, ensemble_i,
                    getattr(self, ensemble_i).to(device, *args, **kwargs))
        return self

    def train(self, *args, **kwargs):
        for i in range(self.num_ensemble):
            getattr(self, 'ensemble_{}'.format(i)).train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        for i in range(self.num_ensemble):
            getattr(self, 'ensemble_{}'.format(i)).eval(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        _dict = {}
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            _dict[name] = getattr(self, name).state_dict(*args, **kwargs)
        return _dict

    def load_state_dict(self, state_dict):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).load_state_dict(state_dict[name])
            setattr(self, name,  getattr(self, name).double())

    def set_obs_bound(self, obs_min, obs_max):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).set_obs_bound(obs_min[i], obs_max[i])

    def set_obs_norm(self, obs_mean, obs_std):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).set_obs_norm(obs_mean, obs_std)

    def set_action_norm(self, action_mean, action_std):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).set_action_norm(action_mean, action_std)

    def set_reward_bound(self, reward_min, reward_max):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).set_reward_bound(reward_min[i], reward_max[i])

    def enable_obs_clip(self):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).enable_obs_clip()

    def enable_reward_clip(self):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).enable_reward_clip()

    def disable_obs_clip(self):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).enable_obs_clip()

    def disable_reward_clip(self):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).disable_reward_clip()

    @property
    def is_obs_clip_enabled(self):
        enabled = True
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            enabled = enabled and getattr(self, name).is_obs_clip_enabled()
            if not enabled:
                break
        return enabled

    @property
    def is_reward_clip_enabled(self):
        enabled = True
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            enabled = enabled and getattr(self, name).is_reward_clip_enabled()
            if not enabled:
                break
        return enabled

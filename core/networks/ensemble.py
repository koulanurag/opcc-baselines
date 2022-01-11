import torch

from .n_step import NstepDynamicsNetwork


class EnsembleDynamicsNetwork:
    def __init__(self, env_name, dataset_name, num_ensemble, obs_size,
                 action_size, hidden_size, n_step, dynamics_type,
                 deterministic=True, constant_prior=False, prior_scale=1.0):
        super().__init__()
        self.__obs_size = obs_size
        self.__action_size = action_size
        self.__num_ensemble = num_ensemble
        self.__deterministic = deterministic
        self.__constant_prior = constant_prior

        for i in range(num_ensemble):
            _net = NstepDynamicsNetwork(env_name=env_name,
                                        dataset_name=dataset_name,
                                        obs_size=obs_size,
                                        action_size=action_size,
                                        hidden_size=hidden_size,
                                        n_step=n_step,
                                        deterministic=deterministic,
                                        dynamics_type=dynamics_type,
                                        constant_prior=constant_prior,
                                        prior_scale=prior_scale)
            setattr(self, 'ensemble_{}'.format(i), _net)

    def reset(self, horizon=1, batch_size=1):
        for i in range(self.num_ensemble):
            getattr(self, 'ensemble_{}'.format(i)).reset(horizon, batch_size)

    def step(self, obs, action):
        assert len(obs.shape) == 3, '(batch , ensemble ,obs. size) required.'
        assert len(action.shape) == 3

        next_obs, reward, done = None, None, None

        for i in range(self.num_ensemble):
            _name = 'ensemble_{}'.format(i)
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

    def update(self, replay_buffer, batch_count: int, batch_size: int):
        loss = {}
        for i in range(self.num_ensemble):
            _name = 'ensemble_{}'.format(i)
            dynamics = getattr(self, _name)
            loss[_name] = dynamics.update(replay_buffer[i], batch_count,
                                          batch_size)
        return loss

    @property
    def num_ensemble(self):
        return self.__num_ensemble

    @property
    def deterministic(self):
        return self.__deterministic

    @property
    def constant_prior(self):
        return self.__constant_prior

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

    def set_obs_bound(self, obs_min, obs_max):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).set_obs_bound(obs_min[i], obs_max[i])

    def set_reward_bound(self, reward_min, reward_max):
        for i in range(self.num_ensemble):
            name = 'ensemble_{}'.format(i)
            getattr(self, name).set_reward_bound(reward_min[i], reward_max[i])

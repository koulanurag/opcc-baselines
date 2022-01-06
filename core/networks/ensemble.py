import torch.nn as nn

from .n_step import NstepDynamicsNetwork


class EnsembleDynamicsNetwork(nn.Module):
    def __init__(self, num_ensemble, obs_size, action_size, hidden_size,
                 n_step, dynamics_type, deterministic=True,
                 constant_prior=False):
        super().__init__()
        self.__obs_size = obs_size
        self.__action_size = action_size
        self.__num_ensemble = num_ensemble
        self.__deterministic = deterministic
        self.__constant_prior = constant_prior

        for i in range(num_ensemble):
            _net = NstepDynamicsNetwork(obs_size, action_size, hidden_size,
                                        n_step, dynamics_type, deterministic,
                                        constant_prior)
            setattr(self, 'ensemble_{}'.format(i), _net)

    def update(self, replay_buffer, batch_count: int, batch_size: int):

        ensemble_loss = {}
        for i in range(self.num_ensemble):
            _name = 'ensemble_{}'.format(i)
            dynamics = getattr(self, _name)
            ensemble_loss[_name] = dynamics.update(replay_buffer[i],
                                                   batch_count,
                                                   batch_size)
        return ensemble_loss

    @property
    def num_ensemble(self):
        return self.__num_ensemble

    @property
    def deterministic(self):
        return self.__deterministic

    @property
    def constant_prior(self):
        return self.__constant_prior

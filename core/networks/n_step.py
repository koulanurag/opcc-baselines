from collections import defaultdict

from .ff import FFDynamicsNetwork


class NstepDynamicsNetwork:
    def __init__(self, obs_size, action_size, hidden_size, n_step=1,
                 dynamics_type='feed-forward', deterministic=True,
                 constant_prior=False, prior_scale=1.0):
        super().__init__()
        self.__n_step = n_step
        for i in range(n_step):
            if dynamics_type == 'feed-forward':
                net = FFDynamicsNetwork(obs_size,
                                        (action_size * (i + 1)),
                                        hidden_size=hidden_size,
                                        constant_prior=constant_prior,
                                        deterministic=deterministic,
                                        activation_function='silu',
                                        prior_scale=prior_scale)
            else:
                raise ValueError()
            setattr(self, 'step_{}'.format(i + 1), net)

    def update(self, replay_buffer, batch_count, batch_size):
        loss = defaultdict(lambda: defaultdict(lambda: 0))
        for batch_i in range(batch_count):
            batch = replay_buffer.sample(batch_size, self.n_step)

            for i in range(self.n_step):
                _name = 'step_{}'.format(i + 1)
                dynamics = getattr(self, _name)

                obs = batch.obs[:, i]
                action = batch.action[:, :i + 1]
                action = action.view(action.shape[0],
                                     action.shape[1] * action.shape[2])
                next_obs = batch.obs[:, i + 1]
                reward = batch.reward[:, i]

                _loss = dynamics.update(obs, action, next_obs, reward)
                for k, v in _loss.items():
                    loss[_name][k] += v

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

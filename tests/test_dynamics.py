import pytest
import torch


@pytest.mark.parametrize('n_step, dynamics_type, constant_prior, deterministic,'
                         ' horizon', [(5, 'feed-forward', True, True, 18),
                                      (1, 'feed-forward', True, True, 18),
                                      (10, 'feed-forward', True, True, 3),
                                      (5, 'autoregressive', True, True, 18),
                                      (1, 'autoregressive', True, True, 18),
                                      (10, 'autoregressive', True, True, 3)])
def test_n_step_forward(n_step, dynamics_type, constant_prior, deterministic,
                        horizon):
    import gym
    from core.networks import EnsembleDynamicsNetwork
    batch_size = 3
    num_ensemble = 5

    env_name = 'HalfCheetah-v2'
    dataset_name = 'random'
    env = gym.make(env_name)
    network = EnsembleDynamicsNetwork(env_name=env_name,
                                      dataset_name=dataset_name,
                                      num_ensemble=num_ensemble,
                                      obs_size=env.observation_space.shape[0],
                                      action_size=env.action_space.shape[0],
                                      hidden_size=50,
                                      n_step=n_step,
                                      deterministic=deterministic,
                                      dynamics_type=dynamics_type,
                                      constant_prior=constant_prior,
                                      prior_scale=0)

    for run in range(3):  # do multiple runs
        network.reset(horizon=horizon, batch_size=batch_size)
        init_obs = [env.observation_space.sample() for _ in range(batch_size)]
        init_obs = torch.tensor(init_obs).unsqueeze(1).float()
        init_obs = init_obs.repeat(1, num_ensemble, 1)
        step_obs = init_obs

        for step_i in range(1, horizon + 1):
            step_action = [env.action_space.sample() for _ in range(batch_size)]
            step_action = torch.tensor(step_action).unsqueeze(1).float()
            step_action = step_action.repeat(1, num_ensemble, 1)
            step_obs, reward, done = network.step(step_obs, step_action)

            if (step_i % n_step) == 0 or step_i == horizon:
                assert (reward != 0).all().item(), \
                    'reward must not be 0 @ step:{}'.format(step_i)
            else:
                assert (reward == 0).all().item(), 'reward must be 0'

        if step_i > n_step:
            with pytest.raises(Exception) as excinfo:
                network.step(step_obs, step_action)
            assert 'cannot step beyond set horizon' in str(excinfo.value)


@pytest.mark.parametrize('obs_min, obs_max, reward_min, reward_max,  '
                         'dynamics_type, deterministic, constant_prior',
                         [(-1, 1, 1, 1, 'feed-forward', True, True),
                          (-1, 1, 1, 1, 'feed-forward', False, True),
                          (-1, 0, 0, 0, 'feed-forward', True, True),
                          (-1, 0, 0, 0, 'feed-forward', False, True),
                          (-2, -1, -2, -1, 'feed-forward', True, True),
                          (-2, -1, -2, -1, 'feed-forward', False, True),
                          (-1, 1, 1, 1, 'autoregressive', True, True),
                          (-1, 1, 1, 1, 'autoregressive', False, True),
                          (-1, 0, 0, 0, 'autoregressive', True, True),
                          (-1, 0, 0, 0, 'autoregressive', False, True),
                          ])
def test_clip(obs_min, obs_max, reward_min, reward_max, dynamics_type,
              deterministic, constant_prior):
    import gym
    from core.networks import EnsembleDynamicsNetwork
    import numpy as np
    batch_size = 3
    num_ensemble = 5
    n_step = 1
    horizon = 10

    env_name = 'HalfCheetah-v2'
    dataset_name = 'random'
    env = gym.make(env_name)
    network = EnsembleDynamicsNetwork(env_name=env_name,
                                      dataset_name=dataset_name,
                                      num_ensemble=num_ensemble,
                                      obs_size=env.observation_space.shape[0],
                                      action_size=env.action_space.shape[0],
                                      hidden_size=50,
                                      n_step=n_step,
                                      deterministic=deterministic,
                                      dynamics_type=dynamics_type,
                                      constant_prior=constant_prior,
                                      prior_scale=1)
    _obs_max = np.ones((network.num_ensemble,
                        env.observation_space.shape[0])) * obs_max
    _obs_min = np.ones((network.num_ensemble,
                        env.observation_space.shape[0])) * obs_min
    _reward_max = np.ones((network.num_ensemble, 1)) * reward_max
    _reward_min = np.ones((network.num_ensemble, 1)) * reward_min
    network.set_reward_bound(_reward_min, _reward_max)
    network.set_obs_bound(_obs_min, _obs_max)
    network.enable_obs_clip()
    network.enable_reward_clip()

    for run in range(3):  # do multiple runs
        network.reset(horizon=horizon, batch_size=batch_size)
        init_obs = [env.observation_space.sample() for _ in range(batch_size)]
        init_obs = torch.tensor(init_obs).unsqueeze(1).float()
        init_obs = init_obs.repeat(1, num_ensemble, 1)
        step_obs = init_obs

        for step_i in range(1, horizon + 1):
            step_action = [env.action_space.sample() for _ in range(batch_size)]
            step_action = torch.tensor(step_action).unsqueeze(1).float()
            step_action = step_action.repeat(1, num_ensemble, 1)
            step_obs, reward, done = network.step(step_obs, step_action)

            assert (reward_min <= reward).all() \
                   and (reward <= reward_max).all(), ' reward out of bounds'
            assert (obs_min <= step_obs).all() \
                   and (step_obs <= obs_max).all(), ' obs. out of bounds'

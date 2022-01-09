import gym
import pytest
import torch

@pytest.mark.parametrize('env_name, dataset_name, n_step, num_ensemble, dynamics_type,horizon',
                         [('Hopper-v2', 'random', 5, 3, 'feed-forward', 20),
                          ('Hopper-v2', 'random', 5, 5, 'feed-forward', 3)])
def test_mc_return(env_name, dataset_name, n_step, num_ensemble, dynamics_type,
                   horizon):
    import cque
    from core.networks import EnsembleDynamicsNetwork
    from core.replay_buffer import ReplayBuffer
    from core.utils import mc_return
    batch_size = 10
    env = gym.make(env_name)
    network = EnsembleDynamicsNetwork(env_name=env_name,
                                      dataset_name=dataset_name,
                                      num_ensemble=num_ensemble,
                                      obs_size=env.observation_space.n,
                                      action_size=env.action_space.n,
                                      hidden_size=200,
                                      n_step=n_step,
                                      deterministic=True,
                                      dynamics_type=dynamics_type,
                                      constant_prior=False,
                                      prior_scale=0)
    policy = lambda x: torch.FloatTensor([env.action_space.sample()
                                          for _ in range(len(x))])
    dataset = cque.get_sequence_dataset(env_name, dataset_name)
    replay_buffer = ReplayBuffer(dataset)

    batch = replay_buffer.sample(n=batch_size, chunk_size=n_step)
    init_obs = dataset[:5]
    predicted_estimates = mc_return(network, init_obs, init_action, policy,
                                    horizon, device='cpu', runs=1,
                                    ensemble_mixture=False,
                                    step_batch_size=128)

    assert predicted_estimates == target_estimates

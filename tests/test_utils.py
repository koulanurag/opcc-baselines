import gym
import numpy as np
import pytest
import torch


class FakeNetwork:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.num_ensemble = 3

    def reset(self, horizon, batch_size, reset_n_step, ensemble_mixture):
        pass

    def step(self, states, actions):
        next_states, rewards, dones = [], [], []
        for batch_i in range(states.shape[0]):
            e_next_states, e_rewards, e_dones = [], [], []
            for ensemble in range(self.num_ensemble):
                state = states[batch_i][ensemble].cpu().numpy()
                state = state.astype('float64')
                action = actions[batch_i][ensemble].cpu().numpy()
                action = action.astype('float32')

                self.env.reset()
                self.env.sim.set_state_from_flattened(state)
                self.env.sim.forward()
                _, reward, done, _ = self.env.step(action)
                next_state = self.env.sim.get_state().flatten()
                e_next_states.append(next_state.tolist())
                e_rewards.append(reward)
                e_dones.append(done)

            next_states.append(e_next_states)
            rewards.append(e_rewards)
            dones.append(e_dones)
        return torch.tensor(next_states).double(),\
               torch.tensor(rewards).double(), \
               torch.tensor(dones)


def test_mc_return():
    import opcc
    from core.utils import mc_return

    env_name = 'Hopper-v2'
    queries = opcc.get_queries(env_name)
    (policy_a_id, policy_b_id) = list(queries.keys())[0]
    query_batch = queries[(policy_a_id, policy_b_id)]
    policy_a, _ = opcc.get_policy(*policy_a_id)

    # query-a
    obss_a = query_batch['info']['state_a']
    actions_a = query_batch['action_a']
    horizons = query_batch['horizon']
    horizon = np.unique(horizons)[0]
    _filter = horizons == horizon
    init_obs = obss_a[_filter]
    init_action = actions_a[_filter]
    fake_network = FakeNetwork(env_name)

    # we receive state and pass on obs. to actor
    def _policy(x):
        # input shape => (batch, ensemble, state)
        return policy_a.actor(x[:, :, 2:])

    predict_1 = mc_return(fake_network, init_obs, init_action, _policy,
                          horizon, device='cpu', runs=10,
                          mixture=False, eval_batch_size=128)
    predict_2 = mc_return(fake_network, init_obs, init_action, _policy,
                          horizon, device='cpu', runs=10, mixture=False,
                          eval_batch_size=128)
    target = query_batch['info']['return_a'][_filter]
    assert (np.round(predict_1, 1) == np.round(predict_2, 1)).all()
    assert np.mean(predict_1.mean(1) - target) < 2, \
        'error between prediction and target is large'


@pytest.mark.parametrize('dynamics_type,prior_scale',
                         [('feed-forward', 0),
                          ('feed-forward', 5),
                          ('autoregressive', 0),
                          ('autoregressive', 5)
                          ])
def test_ensemble_mixture(dynamics_type, prior_scale):
    import opcc
    from core.utils import mc_return
    from core.networks import EnsembleDynamicsNetwork

    env_name = 'Hopper-v2'
    queries = opcc.get_queries(env_name)
    (policy_a_id, policy_b_id) = list(queries.keys())[0]
    query_batch = queries[(policy_a_id, policy_b_id)]
    policy_a, _ = opcc.get_policy(*policy_a_id)

    # query-a
    obss_a = query_batch['obs_a']
    actions_a = query_batch['action_a']
    horizons = query_batch['horizon']
    horizon = np.unique(horizons)[0]
    _filter = horizons == horizon
    init_obs = obss_a[_filter]
    init_action = actions_a[_filter]
    network = EnsembleDynamicsNetwork(env_name=env_name,
                                      dataset_name='xyz',
                                      num_ensemble=20,
                                      obs_size=obss_a[0].shape[0],
                                      action_size=actions_a[0].shape[0],
                                      hidden_size=50,
                                      deterministic=True,
                                      dynamics_type=dynamics_type,
                                      prior_scale=prior_scale)

    for i in range(network.num_ensemble):
        setattr(network, 'ensemble_{}'.format(i),
                getattr(network, 'ensemble_{}'.format(i)).double())

    predict_1 = mc_return(network, init_obs, init_action, policy_a.actor,
                          horizon, device='cpu', runs=10, mixture=True,
                          eval_batch_size=128, mixture_seed=1)
    predict_2 = mc_return(network, init_obs, init_action, policy_a.actor,
                          horizon, device='cpu', runs=10, mixture=True,
                          eval_batch_size=128, mixture_seed=1)

    predict_3 = mc_return(network, init_obs, init_action, policy_a.actor,
                          horizon, device='cpu', runs=10, mixture=True,
                          eval_batch_size=128, mixture_seed=2)

    assert (np.round(predict_1, 1) == np.round(predict_2, 1)).all()
    assert not (np.round(predict_2, 1) == np.round(predict_3, 1)).all()

    target = query_batch['info']['return_a'][_filter]
    assert np.mean(predict_1.mean(1) - target) < 2, \
        'error between prediction and target is large'

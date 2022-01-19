import gym
import numpy as np
import torch
import pytest


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
                state, action = states[batch_i][ensemble], actions[batch_i][ensemble]
                self.env.reset()
                self.env.sim.set_state_from_flattened(state.cpu().numpy())
                self.env.sim.forward()
                _, reward, done, _ = self.env.step(action.cpu().numpy())
                next_state = self.env.sim.get_state().flatten()
                e_next_states.append(next_state.tolist())
                e_rewards.append(reward)
                e_dones.append(done)

            next_states.append(e_next_states)
            rewards.append(e_rewards)
            dones.append(e_dones)
        return torch.tensor(next_states), torch.tensor(rewards), torch.tensor(dones)


def test_mc_return():
    import cque
    from core.utils import mc_return
    import policybazaar

    env_name = 'Hopper-v2'
    queries = cque.get_queries(env_name)
    (policy_a_id, policy_b_id) = list(queries.keys())[0]
    query_batch = queries[(policy_a_id, policy_b_id)]
    policy_a, _ = policybazaar.get_policy(*policy_a_id)

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
        # input shape => (batch,ensemble, state)
        return policy_a.actor(x[:, :, 2:])

    predict_1 = mc_return(fake_network, init_obs, init_action, _policy, horizon,
                          reset_n_step=2, device='cpu', runs=10,
                          ensemble_mixture=False, step_batch_size=128)
    predict_2 = mc_return(fake_network, init_obs, init_action, _policy, horizon,
                          reset_n_step=2, device='cpu', runs=10,
                          ensemble_mixture=False, step_batch_size=128)
    target = query_batch['info']['return_a'][_filter]
    assert (predict_1 == predict_2).all()
    assert np.mean(predict_1.mean(1) - target) < 2, \
        'error between prediction and target is large'


@pytest.mark.parametrize('dynamics_type,constant_prior',
                         [('feed-forward', True),
                          ('feed-forward', False),
                          ('autoregressive', True),
                          ('autoregressive', False)
                          ])
def test_ensemble_mixture(dynamics_type, constant_prior):
    import cque
    from core.utils import mc_return
    from core.networks import EnsembleDynamicsNetwork
    import policybazaar

    env_name = 'Hopper-v2'
    queries = cque.get_queries(env_name)
    (policy_a_id, policy_b_id) = list(queries.keys())[0]
    query_batch = queries[(policy_a_id, policy_b_id)]
    policy_a, _ = policybazaar.get_policy(*policy_a_id)

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
                                      n_step=10,
                                      deterministic=True,
                                      dynamics_type=dynamics_type,
                                      constant_prior=constant_prior,
                                      prior_scale=1)

    predict_1 = mc_return(network, init_obs, init_action, policy_a.actor, horizon,
                          reset_n_step=2,
                          device='cpu', runs=10, ensemble_mixture=True,
                          step_batch_size=128, mixture_seed=1)
    predict_2 = mc_return(network, init_obs, init_action, policy_a.actor, horizon,
                          reset_n_step=2,
                          device='cpu', runs=10, ensemble_mixture=True,
                          step_batch_size=128, mixture_seed=1)

    predict_3 = mc_return(network, init_obs, init_action, policy_a.actor, horizon,
                          reset_n_step=2,
                          device='cpu', runs=10, ensemble_mixture=True,
                          step_batch_size=128, mixture_seed=2)

    assert (predict_1 == predict_2).all()
    assert not (predict_2 == predict_3).all()

    target = query_batch['info']['return_a'][_filter]
    assert np.mean(predict_1.mean(1) - target) < 2, \
        'error between prediction and target is large'

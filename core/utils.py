import logging
import os

import numpy as np
import pandas as pd
import policybazaar
import torch
from rliable import metrics


def init_logger(base_path: str, name: str):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]'
                                  '[%(filename)s>%(funcName)s] => %(message)s')
    file_path = os.path.join(base_path, name + '.log')
    logger = logging.getLogger(name)
    logging.getLogger().handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.FileHandler(file_path, mode='w')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def evaluate_queries(queries, network, runs, batch_size, device='cpu',
                     ensemble_mixture=False):
    predict_df = pd.DataFrame()
    for (policy_a_id, policy_b_id), query_batch in queries.items():

        policy_a, _ = policybazaar.get_policy(*policy_a_id)
        policy_b, _ = policybazaar.get_policy(*policy_b_id)
        policy_a = policy_a.to(device)
        policy_b = policy_b.to(device)

        # query
        obss_a = query_batch['obs_a']
        actions_a = query_batch['action_a']
        obss_b = query_batch['obs_b']
        actions_b = query_batch['action_b']
        horizons = query_batch['horizon']

        predict_a = np.zeros((len(obss_a), network.num_ensemble))
        predict_b = np.zeros((len(obss_b), network.num_ensemble))
        with torch.no_grad():
            for horizon in np.unique(horizons):
                _filter = horizons == horizon
                predict_a[_filter, :] = mc_return(network,
                                                  obss_a[_filter],
                                                  actions_a[_filter],
                                                  policy_a,
                                                  horizon,
                                                  device,
                                                  ensemble_mixture,
                                                  runs,
                                                  batch_size)

                predict_b[_filter, :] = mc_return(network,
                                                  obss_b[_filter],
                                                  actions_b[_filter],
                                                  policy_b,
                                                  horizon,
                                                  device,
                                                  ensemble_mixture,
                                                  runs,
                                                  batch_size)

        for idx in range(len(obss_a)):
            _stat = {
                # ground-truth info
                **{'query_key': (policy_a_id, policy_b_id),
                   'query_idx': idx,
                   'policy_ids': (policy_a_id[1], policy_b_id[1]),
                   'obs_a': obss_a[idx],
                   'action_a': actions_a[idx],
                   'obs_b': obss_b[idx],
                   'action_b': actions_b[idx],
                   'horizon': horizon[idx],
                   'target': query_batch['target'][idx],
                   'returns_a': query_batch['info']['returns_a'][idx],
                   'returns_b': query_batch['info']['returns_b'][idx]},

                # query-a predictions
                **{'pred_a_{}'.format(e_i): predict_a[idx][e_i]
                   for e_i in range(network.num_ensemble)},
                **{'pred_a_mean': predict_a[idx].mean(),
                   'pred_a_iqm': metrics.aggregate_iqm([predict_a[idx]]),
                   'pred_a_median': np.median(predict_a[idx]),
                   'pred_a_max': predict_a[idx].max(),
                   'pred_a_min': predict_a[idx].min()},

                # query-b predictions
                **{'pred_b_{}'.format(e_i): predict_b[idx][e_i]
                   for e_i in range(network.num_ensemble)},
                **{'pred_b_mean': predict_b[idx].mean(),
                   'pred_b_median': np.median(predict_b[idx]),
                   'pred_b_iqm': metrics.aggregate_iqm([predict_b[idx]]),
                   'pred_b_max': predict_b[idx].max(),
                   'pred_b_min': predict_b[idx].min()},
            }
            predict_df = predict_df.append(_stat, ignore_index=True)
    return predict_df


def mc_return(network, init_obs, init_action, policy, horizon: int,
              device='cpu', runs: int = 1, ensemble_mixture: bool = False,
              step_batch_size: int = 128):
    assert len(init_obs) == len(init_action), 'batch size not same'
    batch_size, obs_size = init_obs.shape
    _, action_size = init_action.shape

    # repeat for ensemble size
    init_obs = torch.FloatTensor(init_obs)
    init_obs = init_obs.unsqueeze(1).repeat(1, network.num_ensemble, 1)

    init_action = torch.FloatTensor(init_action)
    init_action = init_action.unsqueeze(1).repeat(1, network.num_ensemble, 1)

    # repeat for runs
    init_obs = init_obs.repeat(runs, 1, 1)
    init_action = init_action.repeat(runs, 1, 1)

    returns = np.zeros((batch_size * runs, network.num_ensemble))
    for batch_idx in range(0, returns.shape[0], step_batch_size):
        batch_end_idx = batch_idx + step_batch_size

        # reset
        step_obs = init_obs[batch_idx:batch_end_idx].to(device)
        step_action = init_action[batch_idx:batch_end_idx].to(device)
        network.reset(max_steps=horizon, batch_size=len(step_obs))

        # step
        for step in range(horizon):
            step_obs, reward, done = network.step(step_obs, step_action)
            step_action = policy(step_obs)

            # move to cpu for saving cuda memory
            reward = reward.cpu().detach().numpy()
            returns[batch_idx:batch_end_idx][~done] += reward[~done]

        if device == 'cuda':
            torch.cuda.empty_cache()

    returns = returns.reshape((batch_size, runs, network.num_ensemble))
    returns = returns.mean(1)
    return returns

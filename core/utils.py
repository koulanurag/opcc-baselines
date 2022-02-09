import logging
import os
import traceback
from typing import List

import numpy as np
import opcc
import pandas as pd
import torch
from rliable import metrics
from tqdm import tqdm

def log_traceback(ex, ex_traceback=None):
    if ex_traceback is None:
        ex_traceback = ex.__traceback__
    tb_lines = [line.rstrip('\n') for line in
                traceback.format_exception(ex.__class__, ex, ex_traceback)]
    return tb_lines


def init_logger(base_path: str, name: str, file_mode='w'):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]'
                                  '[%(filename)s>%(funcName)s] => %(message)s')
    file_path = os.path.join(base_path, name + '.log')
    logger = logging.getLogger(name)
    logging.getLogger().handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.FileHandler(file_path, mode=file_mode)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def evaluate_queries(queries, network, runs, batch_size, device: str = 'cpu',
                     mixture: bool = False) -> pd.DataFrame:
    predict_df = pd.DataFrame()
    for (policy_a_id, policy_b_id), query_batch in tqdm(queries.items(),
                                                        desc='Policy Pairs'):

        # get policies
        policy_a, _ = opcc.get_policy(*policy_a_id)
        policy_b, _ = opcc.get_policy(*policy_b_id)
        policy_a = policy_a.to(device)
        policy_b = policy_b.to(device)

        # query
        obs_a = query_batch['obs_a']
        action_a = query_batch['action_a']
        obs_b = query_batch['obs_b']
        action_b = query_batch['action_b']
        horizons = query_batch['horizon']

        # evaluate
        pred_a = np.zeros((len(obs_a), network.num_ensemble))
        pred_b = np.zeros((len(obs_b), network.num_ensemble))
        for horizon in tqdm(np.unique(horizons), desc='Horizons'):
            _filter = horizons == horizon
            pred_a[_filter, :] = mc_return(network=network,
                                           init_obs=obs_a[_filter],
                                           init_action=action_a[_filter],
                                           policy=policy_a.actor,
                                           horizon=horizon,
                                           device=device,
                                           runs=runs,
                                           mixture=mixture,
                                           eval_batch_size=batch_size,
                                           mixture_seed=0)

            pred_b[_filter, :] = mc_return(network=network,
                                           init_obs=obs_b[_filter],
                                           init_action=action_b[_filter],
                                           policy=policy_b.actor,
                                           horizon=horizon,
                                           device=device,
                                           runs=runs,
                                           mixture=mixture,
                                           eval_batch_size=batch_size,
                                           mixture_seed=0)

        # store in dataframe
        for idx in range(len(obs_a)):
            _stat = {
                # ground-truth info
                **{'env_name': policy_a_id[0],
                   'policy_a_id': policy_a_id[1],
                   'policy_b_id': policy_b_id[1],
                   'query_idx': idx,
                   'policy_ids': (policy_a_id[1], policy_b_id[1]),
                   'obs_a': obs_a[idx],
                   'action_a': action_a[idx],
                   'obs_b': obs_b[idx],
                   'action_b': action_b[idx],
                   'horizon': horizons[idx],
                   'target': query_batch['target'][idx],
                   'return_a': query_batch['info']['return_a'][idx],
                   'return_b': query_batch['info']['return_b'][idx]},

                # query-a predictions
                **{'pred_a_{}'.format(e_i): pred_a[idx][e_i]
                   for e_i in range(network.num_ensemble)},
                **{'pred_a_mean': pred_a[idx].mean(),
                   'pred_a_iqm': metrics.aggregate_iqm([pred_a[idx]]),
                   'pred_a_median': np.median(pred_a[idx]),
                   'pred_a_max': pred_a[idx].max(),
                   'pred_a_min': pred_a[idx].min()},

                # query-b predictions
                **{'pred_b_{}'.format(e_i): pred_b[idx][e_i]
                   for e_i in range(network.num_ensemble)},
                **{'pred_b_mean': pred_b[idx].mean(),
                   'pred_b_median': np.median(pred_b[idx]),
                   'pred_b_iqm': metrics.aggregate_iqm([pred_b[idx]]),
                   'pred_b_max': pred_b[idx].max(),
                   'pred_b_min': pred_b[idx].min()},
            }
            predict_df = predict_df.append(_stat, ignore_index=True)
    return predict_df


@torch.no_grad()
def mc_return(network, init_obs, init_action, policy, horizon: int,
              device: str = 'cpu', runs: int = 1, mixture: bool = False,
              eval_batch_size: int = 128,
              mixture_seed: int = 0) -> List[float]:
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
    for batch_idx in tqdm(range(0, returns.shape[0], eval_batch_size),
                          desc='Batch'):
        batch_end_idx = batch_idx + eval_batch_size

        # reset
        step_obs = init_obs[batch_idx:batch_end_idx].to(device)
        step_action = init_action[batch_idx:batch_end_idx].to(device)

        if mixture:
            network.enable_mixture(mixture_seed)

        # step
        for step in range(horizon):
            step_obs, reward, done = network.step(step_obs, step_action)
            assert len(step_obs.shape) == 3, 'expected (batch, ensemble, obs)'
            step_action = policy(step_obs)
            assert len(step_action.shape) == 3, \
                'expected (batch, ensemble,action)'

            # move to cpu for saving cuda memory
            reward = reward.cpu().detach().numpy()
            returns[batch_idx:batch_end_idx][~done] += reward[~done]

        if device == 'cuda':
            torch.cuda.empty_cache()

        if mixture:
            network.disable_mixture()

    returns = returns.reshape((batch_size, runs, network.num_ensemble))
    returns = returns.mean(1)
    return returns

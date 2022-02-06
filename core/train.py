import logging

import numpy as np
import opcc
import torch
import wandb
from tqdm import tqdm

from core.config import BaseConfig
from core.replay_buffer import ReplayBuffer
from core.utils import init_logger


def train_dynamics(config: BaseConfig):
    # create logger
    init_logger(config.logs_dir_path, 'train_dynamics',
                file_mode='a' if config.args.resume is not None else 'w')

    # download and normalize dataset
    dataset = opcc.get_qlearning_dataset(config.args.env_name,
                                         config.args.dataset_name)
    dataset_size = len(dataset['observations'])

    if config.args.normalize:
        obs_mean = dataset['observations'].mean(axis=0).tolist()
        obs_std = (dataset['observations'].std(axis=0) + 1e-5).tolist()
        action_mean = dataset['actions'].mean(axis=0).tolist()
        action_std = (dataset['actions'].std(axis=0) + 1e-5).tolist()

        dataset['observations'] = ((dataset['observations'] - obs_mean)
                                   / obs_std)
        dataset['next_observations'] = ((dataset['next_observations']
                                         - obs_mean) / obs_std)
        dataset['actions'] = (dataset['actions'] - action_mean) / action_std
    else:
        obs_mean = np.zeros(dataset['observations'][0].shape).tolist()
        obs_std = np.ones(dataset['observations'][0].shape).tolist()
        action_mean = np.zeros(dataset['actions'][0].shape).tolist()
        action_std = np.ones(dataset['actions'][0].shape).tolist()

    # create replay buffers
    replay_buffers = {}
    obs_min, obs_max = [], []
    reward_min, reward_max = [], []
    for ensemble_i in tqdm(range(config.args.num_ensemble),
                           desc="bootstrap sampling"):
        # bootstrap sampling
        idxs = np.random.randint(0, dataset_size, size=dataset_size)

        # get data bounds for clipping during evaluation
        observations = dataset['observations'][idxs]
        obs_min.append(observations.min(axis=0).tolist())
        obs_max.append(observations.max(axis=0).tolist())
        rewards = dataset['rewards'][idxs]
        reward_min.append(rewards.min(axis=0).tolist())
        reward_max.append(rewards.max(axis=0).tolist())

        replay_buffers[ensemble_i] = ReplayBuffer(dataset, idxs, config.device)

    # create and setup network
    network = config.get_uniform_dynamics_network()
    start_count = 0
    if config.args.resume:
        state_dict = torch.load(config.checkpoint_path, torch.device('cpu'))
        network.load_state_dict(state_dict['network'])
        start_count = state_dict['update'] + config.args.log_interval
    else:
        network.set_obs_bound(obs_min, obs_max)
        network.set_reward_bound(reward_min, reward_max)
        network.set_obs_norm(obs_mean, obs_std)
        network.set_action_norm(action_mean, action_std)
    network = network.to(config.device)
    network.train()

    # train
    for update_i in range(start_count, config.args.update_count + 1,
                          config.args.log_interval):
        # estimate ensemble loss and update
        loss = network.update(replay_buffers=replay_buffers,
                              update_count=config.args.log_interval,
                              batch_size=config.args.dynamics_batch_size)
        # log
        _msg = '#{:<10}'.format(update_i)
        for k1, v1 in loss.items():
            for k2, v2 in v1.items():
                _msg += '{}/{} loss:{:<8.3f}'.format(k1, k2, v2)
        logging.getLogger('train_dynamics').info(_msg)
        if config.args.use_wandb:
            wandb.log({**{'update': update_i}, **loss})

        # save
        torch.save({'network': network.state_dict(),
                    'update': update_i}, config.checkpoint_path)
        if config.args.use_wandb:
            wandb.save(glob_str=config.checkpoint_path, policy='now')

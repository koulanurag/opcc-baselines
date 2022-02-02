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
    init_logger(config.logs_dir_path, 'train_dynamics')

    # create network
    network = config.get_uniform_dynamics_network()
    network.train()

    # replay buffers
    dataset = opcc.get_qlearning_dataset(config.args.env_name,
                                         config.args.dataset_name)

    replay_buffers = {}
    obs_min, obs_max = [], []
    reward_min, reward_max = [], []
    for ensemble_i in tqdm(range(network.num_ensemble)):
        # bootstrap sampling
        idxs = np.random.randint(0, len(dataset['observations']),
                                 size=len(dataset['observations']))
        _dataset = {k: v[idxs] for k, v in dataset.items()}
        replay_buffers[ensemble_i] = ReplayBuffer(_dataset, config.device)

        # get data bounds for clipping during evaluation
        observations = _dataset['observations']
        obs_min.append(observations.min(axis=0).tolist())
        obs_max.append(observations.max(axis=0).tolist())

        rewards = _dataset['rewards']
        reward_min.append(rewards.min(axis=0).tolist())
        reward_max.append(rewards.max(axis=0).tolist())

    # setup network
    network.set_obs_bound(obs_min, obs_max)
    network.set_reward_bound(reward_min, reward_max)
    network = network.to(config.device)

    # train
    for update_i in range(0, config.args.update_count + 1,
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

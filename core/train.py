import logging

import opcc
import torch
import wandb

from core.config import BaseConfig
from core.replay_buffer import ReplayBuffer
from core.utils import init_logger
import numpy as np
from tqdm import tqdm


def train_dynamics(config: BaseConfig):
    # create logger
    init_logger(config.logs_dir_path, 'train_dynamics')

    # create network
    network = config.get_uniform_dynamics_network()
    network.train()

    # create replay buffers with BOOTSTRAP SAMPLING
    dataset = opcc.get_qlearning_dataset(config.args.env_name,
                                         config.args.dataset_name)

    replay_buffers = {}
    obs_mins, obs_maxs = [], []
    reward_mins, reward_maxs = [], []
    for ensemble_i in tqdm(range(network.num_ensemble)):
        _idxs = np.random.randint(0, len(dataset), size=len(dataset))
        _dataset = {k: v[_idxs] for k, v in dataset.items()}
        replay_buffers[ensemble_i] = ReplayBuffer(_dataset, config.device)

        # get data bounds for clipping during evaluation
        observations = _dataset['observations']
        obs_min = observations.min(axis=0).tolist()
        obs_max = observations.max(axis=0).tolist()
        rewards = _dataset['rewards']
        reward_min = rewards.min(axis=0).tolist()
        reward_max = rewards.max(axis=0).tolist()

        obs_mins.append(obs_min)
        obs_maxs.append(obs_max)
        reward_mins.append(reward_min)
        reward_maxs.append(reward_max)

    # setup network
    network.set_obs_bound(obs_mins, obs_maxs)
    network.set_reward_bound(reward_mins, reward_maxs)
    network = network.to(config.device)

    # train
    for update_i in range(0, config.args.update_count + 1,
                          config.args.log_interval):
        # estimate ensemble loss and update
        loss: dict = network.update(replay_buffers,
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

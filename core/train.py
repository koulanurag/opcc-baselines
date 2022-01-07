import logging
import random

import cque
import torch
import wandb

from core.config import BaseConfig
from core.replay_buffer import ReplayBuffer
from core.utils import init_logger


def train_dynamics(config: BaseConfig):
    # create logger
    init_logger(config.logs_dir_path, 'train_dynamics')

    # create network
    network = config.get_uniform_dynamics_network().to(config.device)
    network.train()

    # create replay buffers with bootstrap sampling
    dataset = cque.get_sequence_dataset(config.args.env_name,
                                        config.args.dataset_name)
    replay_buffers = {}
    for ensemble_i in range(network.num_ensemble):
        _dataset = random.choices(dataset, k=len(dataset))
        replay_buffers[ensemble_i] = ReplayBuffer(_dataset, config.device)

    # train
    for update_i in range(0, config.args.update_count,
                          config.args.log_interval):
        # estimate ensemble loss and update
        loss: dict = network.update(replay_buffers,
                                    batch_count=config.args.log_interval,
                                    batch_size=config.args.dynamics_batch_size)
        # log
        if config.args.log_interval:
            _msg = '#{:<10}'.format(update_i)
            for k1, v1 in loss.items():
                for k2, v2 in v1.items():
                    for k3, v3 in v2.items():
                        _msg += '{}/{}/{} loss:{:<8.3f}'.format(k1, k2, k3, v3)
            logging.getLogger('train_dynamics').info(_msg)
            if config.args.use_wandb:
                wandb.log({**{'update': update_i}, **loss})

        # save
        if ((update_i % config.args.dynamics_checkpoint_interval == 0)
                or update_i == (config.args.update_count - 1)):
            torch.save({'network': network.state_dict(),
                        'update': update_i}, config.checkpoint_path)
            if config.args.use_wandb:
                wandb.save(glob_str=config.checkpoint_path, policy='now')

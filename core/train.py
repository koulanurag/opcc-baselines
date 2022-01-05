import logging
from collections import defaultdict

import cque
import policybazaar
import torch
import wandb
from core.replay_buffer import ReplayBuffer
from core.test import test_policy

from core.config import BaseConfig
from core.utils import init_logger


def train_dynamics(config: BaseConfig):
    # create logger
    init_logger(config.dynamics_logs_dir_path, 'train_dynamics')
    init_logger(config.dynamics_logs_dir_path, 'test_dynamics')

    # create networks
    dynamics_network = config.get_uniform_dynamics_network().to(config.args.device)
    dynamics_network.train()

    # create dataset
    dataset = cque.get_dataset(config.args.env_name, config.args.dataset_name)
    train_dataset = {k: v for k, v in dataset.items()
                     if k in ['observations', 'rewards', 'actions', 'terminals']}
    if config.args.normalize_obs:
        obs_min, obs_max = train_dataset['observations'].min(axis=0), train_dataset['observations'].max(axis=0)
        ensemble_dynamics_network.set_obs_transform_params(obs_min, obs_max)
        train_dataset['observations'] = ensemble_dynamics_network.transform_obs(train_dataset['observations'])
    if config.args.normalize_reward:
        reward_min, reward_max = train_dataset['rewards'].min(axis=0), train_dataset['rewards'].max(axis=0)
        ensemble_dynamics_network.set_reward_transform_params(reward_min, reward_max)
        train_dataset['rewards'] = ensemble_dynamics_network.transform_reward(train_dataset['rewards'])
    if config.args.normalize_action:
        action_min, action_max = train_dataset['actions'].min(axis=0), train_dataset['actions'].max(axis=0)
        ensemble_dynamics_network.set_action_transform_params(action_min, action_max)
        train_dataset['actions'] = ensemble_dynamics_network.transform_action(train_dataset['actions'])

    if config.args.clip_obs:
        _min, _max = train_dataset['observations'].min(axis=0), train_dataset['observations'].max(axis=0)
        ensemble_dynamics_network.set_obs_bounds(_min, _max)

    if config.args.clip_reward:
        _min, _max = train_dataset['rewards'].min(axis=0), train_dataset['rewards'].max(axis=0)
        ensemble_dynamics_network.set_reward_bounds(_min, _max)

    replay_buffer = {}
    for ensemble_i in range(ensemble_dynamics_network.num_ensemble):
        # Todo : sample
        replay_buffer[ensemble_i] = ReplayBuffer({k: v for k, v in train_dataset.items()},
                                                 config.args.device, store_terminal_prob=config.args.train_terminal)

    # create envs with neural dynamics to test with base-policies
    neural_env = config.neural_env(dataset['observations'], ensemble_dynamics_network,
                                   num_envs=config.args.dynamics_test_episodes)

    # load baseline policy
    base_policies = {}
    for i in range(1, 5):
        base_policy = policybazaar.get_policy(config.args.env_name, pre_trained=i)[0].to(config.args.device)
        base_policies['level_{}'.format(i)] = base_policy

    # train
    for epoch in range(config.args.num_epochs):

        # estimate ensemble loss and update
        epoch_loss = defaultdict(lambda: 0)
        wandb_log_items = {}
        for ensemble_i, optimizer in dynamics_optimizer.items():
            loss = update_params(config, ensemble_dynamics_network.ensemble(ensemble_i),
                                 optimizer, replay_buffer[ensemble_i])

            for key, value in loss.items():
                epoch_loss[key] += value
                wandb_log_items['train_dynamics_{}/{}_loss'.format(ensemble_i, key)] = value

        epoch_loss = {key: value / ensemble_dynamics_network.num_ensemble
                      for key, value in epoch_loss.items()}
        wandb_log_items = {**wandb_log_items,
                           **{'train_dynamics/{}_loss'.format(key): value
                              for key, value in epoch_loss.items()}}

        # Log to File
        _msg = 'Epoch #{:<10} '.format(epoch)
        _msg += ' '.join('{} loss : {:<8.3f}'.format(key, value)
                         for key, value in epoch_loss.items())
        logging.getLogger('train_dynamics').info(_msg)

        # Test dynamics
        if (epoch + 1) % config.args.dynamics_test_interval == 0:

            # test policies
            policy_info = {}
            for base_policy_name, base_policy in base_policies.items():
                _info = test_policy(neural_env, base_policy, device=config.args.device)
                test_score_array = _info['episode_reward'].sum(axis=0).mean()
                policy_info['{}-score-mean'.format(base_policy_name)] = test_score_array

            # Log
            _msg = ''
            for test_name, test_value in policy_info.items():
                _msg += ' {}: {}'.format(test_name, test_value)
                wandb_log_items['test_dynamics/{}'.format(test_name)] = test_value
            logging.getLogger('test_dynamics').info(_msg)

        # Log to wandb
        if config.args.use_wandb:
            wandb.log({**{'epoch': epoch}, **wandb_log_items})

        # save
        if (epoch % config.args.dynamics_checkpoint_interval == 0) \
                or epoch == (config.args.num_epochs - 1):
            torch.save(dynamics_network.state_dict(),
                       config.dynamics_network_path)
            torch.save({'network': dynamics_network.state_dict(),
                        'optimizer': {k: v.state_dict()
                                      for k, v in dynamics_optimizer.items()},
                        'epoch_i': epoch},
                       config.dynamics_checkpoint_path)
            if config.args.use_wandb:
                wandb.save(glob_str=config.dynamics_checkpoint_path,
                           policy='now')
                wandb.save(glob_str=config.dynamics_network_path,
                           policy='now')

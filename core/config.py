import hashlib
import os
from collections import defaultdict

import gym

from core.networks import EnsembleDynamicsNetwork

ACTION_SCALE = defaultdict(lambda: defaultdict(lambda: 1))


class BaseConfig(object):
    """
    Base configuration for experiments
    """
    def __init__(self, args, dynamics_args):
        self.__args = args

        # dynamics hyper-parameters hash for experiment saving
        sorted_dyn_args = sorted(dynamics_args._group_actions,
                                 key=lambda x: x.dest)
        dyn_args_str = [str(vars(args)[hp.dest]) for hp in sorted_dyn_args]
        dyn_args_byte = bytes(''.join(dyn_args_str), 'ascii')
        dyn_hp_hash = hashlib.sha224(dyn_args_byte).hexdigest()

        base_path = os.path.join(args.result_dir, args.env_name,
                                 args.dataset_name)
        self.exp_dir_path = os.path.join(base_path, dyn_hp_hash)
        self.exp_dir_path = os.path.abspath(self.exp_dir_path)

        # create directories
        os.makedirs(self.exp_dir_path, exist_ok=True)
        os.makedirs(self.logs_dir_path, exist_ok=True)

        # store env attributes
        env = self.new_game()
        self.observation_size = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.action_space_low = env.action_space.low
        self.action_space_high = env.action_space.high
        self.action_size = env.action_space.shape[0]
        self.max_episode_steps = env.unwrapped.spec.max_episode_steps

    def get_uniform_dynamics_network(self) -> EnsembleDynamicsNetwork:
        return EnsembleDynamicsNetwork(env_name=self.args.env_name,
                                       dataset_name=self.args.dataset_name,
                                       num_ensemble=self.args.num_ensemble,
                                       obs_size=self.observation_size,
                                       action_size=self.action_size,
                                       hidden_size=self.args.hidden_size,
                                       deterministic=self.args.deterministic,
                                       dynamics_type=self.args.dynamics_type,
                                       prior_scale=self.args.constant_prior_scale)

    def new_game(self):
        return gym.make('{}'.format(self.args.env_name))

    @property
    def args(self):
        return self.__args

    @property
    def network_path(self) -> os.path:
        return os.path.join(self.exp_dir_path, 'dynamics_network.p')

    def evaluate_queries_path(self, args, queries_args) -> os.path:
        # hyper-parameters hash for experiment saving
        sorted_args = sorted(queries_args._group_actions,
                             key=lambda x: x.dest)
        query_args_str = [str(vars(args)[hp.dest]) for hp in sorted_args]
        query_args_byte = bytes(''.join(query_args_str), 'ascii')
        query_args_hash = hashlib.sha224(query_args_byte).hexdigest()
        _dir = os.path.join(self.exp_dir_path, query_args_hash)
        os.makedirs(_dir, exist_ok=True)
        return os.path.join(_dir, 'evaluate_queries.pkl')

    def uncertainty_exp_dir(self, args, queries_args,
                            uncertainty_args) -> os.path:
        # hyper-parameters hash for experiment saving
        sorted_args = sorted(queries_args._group_actions,
                             key=lambda x: x.dest)
        query_args_str = [str(vars(args)[hp.dest]) for hp in sorted_args]
        query_args_byte = bytes(''.join(query_args_str), 'ascii')
        query_args_hash = hashlib.sha224(query_args_byte).hexdigest()

        # hyper-parameters hash for experiment saving
        sorted_args = sorted(uncertainty_args._group_actions,
                             key=lambda x: x.dest)
        uncertainty_args_str = [str(vars(args)[hp.dest]) for hp in sorted_args]
        uncertainty_args_byte = bytes(''.join(uncertainty_args_str), 'ascii')
        uncertainty_args_hash = hashlib.sha224(uncertainty_args_byte)
        uncertainty_args_hash = uncertainty_args_hash.hexdigest()

        _dir = os.path.join(self.exp_dir_path, query_args_hash,
                            uncertainty_args_hash)
        os.makedirs(_dir, exist_ok=True)

        return _dir

    @property
    def checkpoint_path(self) -> os.path:
        return os.path.join(self.exp_dir_path, 'dynamics_checkpoint.p')

    @property
    def logs_dir_path(self) -> os.path:
        return os.path.join(self.exp_dir_path, 'dynamics_logs')

    @property
    def device(self) -> str:
        return self.args.device

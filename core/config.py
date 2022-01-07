import hashlib
import os
from collections import defaultdict

import gym
import numpy as np

# from core.neural_env import NeuralEnv
#
from core.networks import EnsembleDynamicsNetwork

ACTION_SCALE = defaultdict(lambda: defaultdict(lambda: 1))


class BaseConfig(object):
    def __init__(self, args, dynamics_args):
        self.__args = args

        # hyper-parameters hash
        dynamics_args_str = [str(vars(args)[hp.dest])
                             for hp in sorted(dynamics_args._group_actions,
                                              key=lambda x: x.dest)]
        dyn_hp_hash = hashlib.sha224(
            bytes(''.join(sorted(dynamics_args_str)), 'ascii')).hexdigest()

        base_path = os.path.join(args.result_dir, args.env_name,
                                 args.dataset_name)
        self.dynamics_exp_dir_path = os.path.join(base_path,
                                                  'dynamic:{}'.format(
                                                      dyn_hp_hash))
        self.dynamics_exp_dir_path = os.path.abspath(
            self.dynamics_exp_dir_path)

        # create directories
        os.makedirs(self.dynamics_exp_dir_path, exist_ok=True)
        os.makedirs(self.dynamics_logs_dir_path, exist_ok=True)

        # store env attributes
        env = self.new_game()
        self.observation_size = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.action_space_low = env.action_space.low
        self.action_space_high = env.action_space.high
        self.action_size = env.action_space.shape[0]
        self.max_episode_steps = env.unwrapped.spec.max_episode_steps

    def sample_uniform_action(self, n=1):
        return np.random.uniform(self.action_space_low, self.action_space_high,
                                 size=(n, len(self.action_space_low))).tolist()

    def get_uniform_dynamics_network(self):
        return EnsembleDynamicsNetwork(num_ensemble=self.args.num_ensemble,
                                       obs_size=self.observation_size,
                                       action_size=self.action_size,
                                       hidden_size=self.args.hidden_size,
                                       n_step=self.args.n_step_model,
                                       deterministic=self.args.deterministic,
                                       dynamics_type=self.args.dynamics_type,
                                       constant_prior=self.args.constant_prior,
                                       prior_scale=self.args.prior_scale)

    def new_game(self):
        return gym.make('{}'.format(self.args.env_name))

    def new_vectorized_game(self, num_envs=1):
        from gym.vector.sync_vector_env import SyncVectorEnv
        envs = SyncVectorEnv([self.new_game for _ in range(num_envs)])
        return envs

    def neural_env(self, init_observations, ensemble_dynamics_network,
                   num_envs=1):
        envs = self.new_vectorized_game(num_envs)
        return NeuralEnv(self.__args.env_name, init_observations,
                         ensemble_dynamics_network,
                         envs.action_space, self.max_episode_steps,
                         num_env=num_envs,
                         device=self.__args.device)

    @property
    def args(self):
        return self.__args

    @property
    def action_scale(self):
        return ACTION_SCALE[self.__args.case][self.__args.env_name]

    @property
    def dynamics_network_path(self):
        return os.path.join(self.dynamics_exp_dir_path, 'dynamics_network.p')

    @property
    def dynamics_checkpoint_path(self):
        return os.path.join(self.dynamics_exp_dir_path,
                            'dynamics_checkpoint.p')

    @property
    def dynamics_logs_dir_path(self):
        return os.path.join(self.dynamics_exp_dir_path, 'dynamics_logs')

    @property
    def device(self):
        return self.__args.device

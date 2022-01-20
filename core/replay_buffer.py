import random
import warnings
from typing import NamedTuple

import numpy as np
import torch


class BatchOutput(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor
    timeout: torch.Tensor


class ReplayBuffer:
    def __init__(self, sequence_dataset, device='cpu'):
        self.dataset = {k: [] for k in sequence_dataset[0]}
        self.seq_lens = []
        for seq in sequence_dataset:
            self.seq_lens.append(len(seq['rewards']))
            for k, v in seq.items():
                # if self.dataset[k] is not None:
                    # self.dataset[k] = np.concatenate((self.dataset[k], v),
                    #                                  axis=0)
                self.dataset[k].append(v)
                # else:
                #     self.dataset[k] = [v]
        self.dataset['observations'] = np.concatenate(self.dataset['observations'],0)
        self.dataset['actions'] = np.concatenate(self.dataset['actions'],0)
        self.dataset['rewards'] = np.concatenate(self.dataset['rewards'],0)
        self.dataset['timeouts'] = np.concatenate(self.dataset['timeouts'],0)
        self.dataset['terminals'] = np.concatenate(self.dataset['terminals'],0)

        self.__obs_size = self.dataset['observations'][0].shape[0]
        self.__action_size = self.dataset['actions'][0].shape[0]
        self.device = device

    # @profile
    def sample(self, n: int, chunk_size: int) -> BatchOutput:
        if chunk_size == 1:
            return self.sample_1_step(n, chunk_size)
        else:
            return self.sample_n_step(n, chunk_size)

        return BatchOutput(obs, action, reward, terminal, timeout)

    def sample_1_step(self, n, chunk_size):
        assert n >= 1, 'batch size should be at least 1'
        assert chunk_size == 1, 'chunk size should be at least 1'

        start_idxs = np.random.randint(low=0, high=self.size - chunk_size,
                                       size=n)
        obs = np.concatenate((np.expand_dims(self.dataset['observations'][start_idxs], axis=1),
                              np.expand_dims(self.dataset['observations'][start_idxs + 1], axis=1)), axis=1)
        action = np.expand_dims(self.dataset['actions'][start_idxs],axis=1)
        reward = np.expand_dims(self.dataset['rewards'][start_idxs], axis=1)
        terminal = np.expand_dims(self.dataset['terminals'][start_idxs], axis=1)
        timeout = np.expand_dims(self.dataset['timeouts'][start_idxs], axis=1)

        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.float)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        terminal = torch.tensor(terminal, device=self.device, dtype=torch.float)
        timeout = torch.tensor(timeout, device=self.device, dtype=torch.float)

        return BatchOutput(obs, action, reward, terminal, timeout)

    def sample_n_step(self, n: int, chunk_size: int) -> BatchOutput:
        assert n >= 1, 'batch size should be at least 1'
        assert chunk_size > 1, 'chunk size should be at least 1'

        # sample batch
        obs = np.empty((n, chunk_size + 1, self.__obs_size))
        action = np.empty((n, chunk_size, self.__action_size))
        reward = np.empty((n, chunk_size))
        terminal = np.empty((n, chunk_size))
        timeout = np.empty((n, chunk_size))
        start_idxs = np.random.randint(low=0, high=self.size - chunk_size,
                                       size=n)

        for batch_i, start_idx in enumerate(start_idxs):
            end_idx = start_idx + chunk_size

            obs[batch_i, :] = self.dataset['observations'][start_idx:end_idx + 1]
            action[batch_i, :] = self.dataset['actions'][start_idx:end_idx]
            reward[batch_i, :] = self.dataset['rewards'][start_idx:end_idx]
            terminal[batch_i, :] = self.dataset['terminals'][start_idx:end_idx]
            timeout[batch_i, :] = self.dataset['timeouts'][start_idx:end_idx]

        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.float)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        terminal = torch.tensor(terminal, device=self.device, dtype=torch.float)
        timeout = torch.tensor(timeout, device=self.device, dtype=torch.float)

        return BatchOutput(obs, action, reward, terminal, timeout)

    @property
    def size(self):
        return len(self.dataset['rewards'])

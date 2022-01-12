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
    def __init__(self, sequence_dataset, chunk_size, device='cpu',
                 suppress_warnings: bool = False):
        assert chunk_size >= 1, 'chunk size should be at least 1'
        self.__chunk_size = chunk_size
        self.dataset = np.array([{**seq, **{'step_count': len(seq['rewards'])}}
                                 for seq in sequence_dataset
                                 if len(seq['rewards']) > chunk_size])
        # check for short sequences
        if self.size == 0:
            raise Exception('no sequence of chunk-size {}'.format(chunk_size))
        elif 0 < self.size < len(sequence_dataset):
            if not suppress_warnings:
                warnings.warn(": only {} out of {} are considered for"
                              " sampling".format(self.size,
                                                 len(sequence_dataset)))

        self.__obs_size = self.dataset[0]['observations'][0].shape[0]
        self.__action_size = self.dataset[0]['actions'][0].shape[0]
        self.device = device

    def sample(self, n: int):
        assert n >= 1, 'batch size should be at least 1'
        # sample batch
        obs = np.empty((n, self.chunk_size + 1, self.__obs_size))
        action = np.empty((n, self.chunk_size, self.__action_size))
        reward = np.empty((n, self.chunk_size))
        terminal = np.empty((n, self.chunk_size))
        timeout = np.empty((n, self.chunk_size))
        seq_idxs = np.random.randint(low=0, high=self.size, size=n)

        for batch_i in range(n):
            seq = self.dataset[seq_idxs[batch_i]]
            seq_size = seq['step_count']
            start_i = np.random.randint(0,  seq_size - self.chunk_size)
            end_i = start_i + self.chunk_size

            obs[batch_i, :] = seq['observations'][start_i:end_i + 1]
            action[batch_i, :] = seq['actions'][start_i:end_i]
            reward[batch_i, :] = seq['rewards'][start_i:end_i]
            terminal[batch_i, :] = seq['terminals'][start_i:end_i]
            timeout[batch_i, :] = seq['timeouts'][start_i:end_i]

        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.float)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        terminal = torch.tensor(terminal, device=self.device, dtype=torch.float)
        timeout = torch.tensor(timeout, device=self.device, dtype=torch.float)

        return BatchOutput(obs, action, reward, terminal, timeout)

    @property
    def size(self):
        return len(self.dataset)

    @property
    def chunk_size(self):
        return self.__chunk_size

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
        self.dataset = np.array(sequence_dataset)
        self.__seq_len = np.array([len(seq['rewards'])
                                   for seq in sequence_dataset])
        self.__obs_size = self.dataset[0]['observations'][0].shape[0]
        self.__action_size = self.dataset[0]['actions'][0].shape[0]
        self.device = device

    def sample(self, n: int, chunk_size: int, suppress_warnings: bool = False):
        assert n >= 1, 'batch size should be at least 1'
        assert chunk_size >= 1, 'chunk size should be at least 1'
        valid_seq = self.__seq_len > chunk_size

        # check for short sequences
        if sum(valid_seq) == 0:
            raise Exception('no sequence of chunk-size {}'.format(chunk_size))
        elif sum(valid_seq) < self.size:
            if not suppress_warnings:
                warnings.warn(": only {} out of {} are considered for"
                              " sampling".format(sum(valid_seq), self.size))

        # sample batch
        obs = torch.empty((n, chunk_size + 1, self.__obs_size))
        action = torch.empty((n, chunk_size, self.__action_size))
        reward = torch.empty((n, chunk_size), dtype=float)
        terminal = torch.empty((n, chunk_size), dtype=bool)
        timeout = torch.empty((n, chunk_size), dtype=bool)

        for batch_i in range(n):
            seq = random.choice(self.dataset[valid_seq])
            seq_size = len(seq['rewards'])
            start_i = random.randint(0, seq_size - chunk_size - 1)
            end_i = start_i + chunk_size

            obs[batch_i, :] = torch.tensor(seq['observations'][start_i:end_i + 1])
            action[batch_i, :] = torch.tensor(seq['actions'][start_i:end_i])
            reward[batch_i, :] = torch.tensor(seq['rewards'][start_i:end_i])
            terminal[batch_i, :] = torch.tensor(seq['terminals'][start_i:end_i])
            timeout[batch_i, :] = torch.tensor(seq['timeouts'][start_i:end_i])

        obs = obs.to(device=self.device).float()
        action = action.to(device=self.device).float()
        reward = reward.to(device=self.device).float()
        terminal = terminal.to(device=self.device).float()
        timeout = timeout.to(device=self.device).float()

        return BatchOutput(obs, action, reward, terminal, timeout)

    @property
    def size(self):
        return len(self.dataset)

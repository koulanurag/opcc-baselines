import random
from typing import NamedTuple
import torch
import numpy as np
import warnings


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
        obs = []
        action = []
        reward = []
        terminal = []
        timeout = []

        for step_i in range(n):
            seq = random.choice(self.dataset[valid_seq])
            seq_size = len(seq['rewards'])
            start_idx = random.randint(0, seq_size - chunk_size - 1)
            end_idx = start_idx + chunk_size
            obs.append(seq['observations'][start_idx:end_idx + 1])
            reward.append(seq['rewards'][start_idx:end_idx])
            action.append(seq['actions'][start_idx:end_idx])
            terminal.append(seq['terminals'][start_idx:end_idx])
            timeout.append(seq['timeouts'][start_idx:end_idx])

        obs = torch.tensor(obs).to(device=self.device).float()
        reward = torch.tensor(reward).to(device=self.device).float()
        action = torch.tensor(action).to(device=self.device).float()
        terminal = torch.tensor(terminal).to(device=self.device).float()
        timeout = torch.tensor(timeout).to(device=self.device).float()

        return BatchOutput(obs, action, reward, terminal, timeout)

    @property
    def size(self):
        return len(self.dataset)

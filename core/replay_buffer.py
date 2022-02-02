from typing import NamedTuple

import numpy as np
import torch


class BatchOutput(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor


class ReplayBuffer:
    def __init__(self, qlearning_dataset, device='cpu'):
        self.dataset = qlearning_dataset
        self.__obs_size = self.dataset['observations'][0].shape[0]
        self.__action_size = self.dataset['actions'][0].shape[0]
        self.device = device

    def sample(self, n: int) -> BatchOutput:
        idxs = np.random.randint(low=0, high=self.size, size=n)
        obs = self.dataset['observations'][idxs]
        action = self.dataset['actions'][idxs]
        next_obs = self.dataset['observations'][idxs]
        reward = self.dataset['rewards'][idxs]
        terminal = self.dataset['terminals'][idxs]

        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.float)
        next_obs = torch.tensor(next_obs, device=self.device,
                                dtype=torch.float)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        terminal = torch.tensor(terminal, device=self.device,
                                dtype=torch.float)

        return BatchOutput(obs, action, next_obs, reward, terminal)

    @property
    def size(self) -> int:
        return len(self.dataset['rewards'])

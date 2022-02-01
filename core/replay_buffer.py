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


class ReplayBuffer:
    def __init__(self, qlearning_dataset, device='cpu'):
        self.dataset = {'observations': qlearning_dataset['observations'],
                        'actions': qlearning_dataset['actions'],
                        'rewards': qlearning_dataset['rewards'],
                        'terminals': qlearning_dataset['terminals']}

        self.__obs_size = self.dataset['observations'][0].shape[0]
        self.__action_size = self.dataset['actions'][0].shape[0]
        self.device = device

    def sample(self, n: int) -> BatchOutput:
        idxs = np.random.randint(low=0, high=self.size, size=n)
        observations = self.dataset['observations'][idxs]
        rewards = self.dataset['rewards'][idxs]
        actions = self.dataset['actions'][idxs]
        terminals = self.dataset['terminals'][idxs]

        obs = torch.tensor(observations, device=self.device, dtype=torch.float)
        action = torch.tensor(actions, device=self.device, dtype=torch.float)
        reward = torch.tensor(rewards, device=self.device, dtype=torch.float)
        terminal = torch.tensor(terminals, device=self.device,
                                dtype=torch.float)

        return BatchOutput(obs, action, reward, terminal)

    @property
    def size(self) -> int:
        return len(self.dataset['rewards'])

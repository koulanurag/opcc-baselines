import numpy as np
import torch
import torch.nn as nn


def weights_init(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape),
                                                        mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear:
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


def is_terminal(env_name, obs):
    """
    Reference:
    - Page 12: https://arxiv.org/pdf/1604.06778.pdf
    - MBPO original paper code: https://github.com/jannerm/mbpo/tree/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129/mbpo/static
    - MBPO Pytorch paper reference: https://openreview.net/pdf?id=rkezvT9f6r
    """

    if env_name == "Hopper-v2":
        assert len(obs.shape) == 2

        height = obs[:, 0]
        angle = obs[:, 1]
        not_done = (np.isfinite(obs).all(axis=-1)
                    * np.abs(obs[:, 1:] < 100).all(axis=-1)
                    * (height > .7)
                    * (np.abs(angle) < .2))
        not_done = not_done.bool()
        done = ~not_done
        return done
    elif env_name == "Walker2d-v2":
        assert len(obs.shape) == 2

        height = obs[:, 0]
        angle = obs[:, 1]
        not_done = ((height > 0.8)
                    * (height < 2.0)
                    * (angle > -1.0)
                    * (angle < 1.0))
        not_done = not_done.bool()
        done = ~not_done
        return done
    elif 'maze' in env_name or env_name == 'HalfCheetah-v2':
        done = torch.zeros(obs.shape[:1]).bool()
        return done
    else:
        raise ValueError('{} termination rule not found'.format(env_name))

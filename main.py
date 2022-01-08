import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from core.config import BaseConfig
from core.train import train_dynamics


def _seed(seed=0, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def get_args(arg_str: str = None):
    # gather arguments
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='cque-baselines',
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    # job arguments
    job_args = parser.add_argument_group('job args')
    job_args.add_argument('--env-name', default='HalfCheetah-v2',
                          help='name of the environment')
    job_args.add_argument('--dataset-name', default='random',
                          help='name of the dataset')
    job_args.add_argument('--no-cuda', action='store_true',
                          help='no cuda usage')
    job_args.add_argument('--job', required=True,
                          choices=['train-dynamics',
                                   'evaluate-queries',
                                   'uncertainty-test',
                                   'uncertainty-test-plot'])
    # paths
    path_args = parser.add_argument_group('paths setup')
    path_args.add_argument('--d4rl-dataset-dir', type=Path,
                           default=Path(os.path.join('~/.d4rl', 'datasets')),
                           help="directory to store d4rl datasets")
    path_args.add_argument('--policybazaar-dir', type=Path,
                           default=Path(os.path.join('~/.policybazaar')),
                           help="directory to store policybazaar data")
    path_args.add_argument('--result-dir', type=Path,
                           default=Path(os.path.join(os.getcwd(), 'results')),
                           help="directory to store results")
    # wandb setup
    wandb_args = parser.add_argument_group('wandb setup')
    wandb_args.add_argument('--wandb-project-name', default='cque-baselines',
                            help='name of the wandb project')
    wandb_args.add_argument('--use-wandb', action='store_true',
                            help='use Weight and bias visualization lib')
    wandb_args.add_argument('--wandb-dir', default=os.path.join('~/.wandb'),
                            help="directory Path to store wandb data")

    # dynamics args
    dynamics_args = parser.add_argument_group('args for training dynamics')
    dynamics_args.add_argument('--dynamics-type', default='feed-forward',
                               choices=['feed-forward', 'autoregressive'],
                               help='type of dynamics model')
    dynamics_args.add_argument('--deterministic', action='store_true',
                               help='if True, we use deterministic model '
                                    'otherwise stochastic')
    dynamics_args.add_argument('--train-terminal', action='store_true',
                               help='enables training of terminal flag')
    dynamics_args.add_argument('--use-dropout', action='store_true',
                               help='uses dropout within models')
    dynamics_args.add_argument('--n-step-model', default=1, type=int,
                               help='n-step predictor for model ')
    dynamics_args.add_argument('--dynamics-seed', default=0, type=int,
                               help='seed for training dynamics ')
    dynamics_args.add_argument('--log-interval', default=1, type=int,
                               help='epoch interval to test dynamics ')
    dynamics_args.add_argument('--dynamics-test-interval', default=1, type=int,
                               help='epoch interval to test dynamics ')
    dynamics_args.add_argument('--dynamics-checkpoint-interval', type=int,
                               default=1, help='epoch interval to save'
                                               ' dynamics checkpoint ')
    dynamics_args.add_argument('--hidden-size', type=int, default=200,
                               help='hidden size for Linear Layers ')
    dynamics_args.add_argument('--dynamics-test-episodes', type=int, default=5,
                               help='no. of test episodes for evaluating '
                                    'dynamics with base policies')

    dynamics_args.add_argument('--update-count', type=int, default=100,
                               help='epochs for training ')
    dynamics_args.add_argument('--batch-count', type=int, default=20,
                               help='batches per epochs ')
    dynamics_args.add_argument('--dynamics-batch-size', type=int, default=256,
                               help='batch size for Dynamics Learning ')
    dynamics_args.add_argument('--reward-loss-coeff', type=int, default=1,
                               help='reward loss coefficient for training ')
    dynamics_args.add_argument('--observation-loss-coeff', type=int, default=1,
                               help='obs. loss coefficient for training ')
    dynamics_args.add_argument('--terminal-loss-coeff', type=int, default=1,
                               help='terminal loss coefficient for training')
    dynamics_args.add_argument('--grad-clip-norm', type=float, default=5.0,
                               help='gradient clipping norm')
    dynamics_args.add_argument('--dynamics-lr', type=float, default=1e-3,
                               help='learning rate for Dynamics')

    dynamics_args.add_argument('--clip-obs', action='store_true',
                               help='clip the observation space with bounds')
    dynamics_args.add_argument('--clip-reward', action='store_true',
                               help='clip the reward with dataset bounds ')
    dynamics_args.add_argument('--normalize-obs', action='store_true',
                               help='normalizes the observation space ')
    dynamics_args.add_argument('--normalize-reward', action='store_true',
                               help='normalizes the rewards')
    dynamics_args.add_argument('--normalize-action', action='store_true',
                               help='normalizes the action space ')
    dynamics_args.add_argument('--add-observation-noise', action='store_true',
                               help='adds noise to observation')
    dynamics_args.add_argument('--add-reward-noise', action='store_true',
                               help='adds noise to rewards')
    dynamics_args.add_argument('--observation-noise-std', type=float,
                               default=0.01,
                               help='std for adding noise to observation')
    dynamics_args.add_argument('--reward-noise-std', type=float, default=0.01,
                               help='std for adding noise to reward ')
    dynamics_args.add_argument('--num-ensemble', default=1, type=int,
                               help='number of dynamics for ensemble ')
    dynamics_args.add_argument('--constant-prior', action='store_true',
                               help='adds a constant prior to each model ')
    dynamics_args.add_argument('--constant-prior-scale', type=float,
                               default=1,
                               help='scale for constant priors')

    # queries evaluation args
    queries_args = parser.add_argument_group('args for evaluating queries')
    queries_args.add_argument('--restore-dynamics-from-wandb',
                              action='store_true',
                              help='restore model from wandb run')
    queries_args.add_argument('--wandb-dynamics-run-id', type=str,
                              help='wandb run id if restoring model')
    queries_args.add_argument('--query-eval-ensemble-mixture',
                              action='store_true',
                              help='if enabled, mixes ensemble models '
                                   'for evaluation  ')
    queries_args.add_argument('--query-eval-runs', type=int, default=1,
                              help='run count for each query ')
    queries_args.add_argument('--query-eval-step-batch-size', type=int,
                              default=128,
                              help='batch size for query evaluation ')

    # uncertainty-test arguments
    uncertain_args = parser.add_argument_group('args for  uncertainty-test')
    uncertain_args.add_argument('--uncertainty-test-type',
                                default='ensemble-voting',
                                choices=['paired-confidence-interval',
                                         'unpaired-confidence-interval',
                                         'ensemble-voting'],
                                help='type of uncertainty test')
    uncertain_args.add_argument('--wandb-query-data-run-id', type=str,
                                help='wandb query run id ')

    # Process arguments
    args = parser.parse_args(arg_str.split(" ") if arg_str else None)
    args.device = ('cuda' if (not args.no_cuda) and
                             torch.cuda.is_available() else 'cpu')

    return args, job_args, path_args, wandb_args, \
           dynamics_args, queries_args, uncertain_args


if __name__ == '__main__':
    (args, job_args, path_args, wandb_args, dynamics_args,
     queries_args, uncertainty_args) = get_args()

    # d4rl setup
    os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
    os.environ['D4RL_DATASET_DIR'] = str(args.d4rl_dataset_dir)
    os.environ['POLICYBAZAAR_DIR'] = str(args.policybazaar_dir)

    if args.job == 'train-dynamics':
        config = BaseConfig(args, dynamics_args)
        if args.use_wandb:  # used for experiment tracking
            wandb.init(job_type=args.job,
                       dir=args.wandb_dir,
                       project=args.wandb_project_name + '-' + args.job,
                       settings=wandb.Settings(start_method="thread"))
            wandb.config.update({x.dest: vars(args)[x.dest]
                                 for x in job_args._group_actions})
            wandb.config.update({x.dest: vars(args)[x.dest]
                                 for x in dynamics_args._group_actions})
        train_dynamics(config)

    elif args.job == 'evaluate-queries':
        if args.restore_dynamics_from_wandb:
            assert args.wandb_dynamics_run_id is not None, \
                'w&b id cannot be {}'.format(args.wandb_dynamics_run_id)

    else:
        raise NotImplementedError()

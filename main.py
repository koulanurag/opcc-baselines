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
    parser = argparse.ArgumentParser(description='cque-baselines')

    # job arguments
    job_args = parser.add_argument_group('job args')
    job_args.add_argument('--env-name', default='HalfCheetah-v2')
    job_args.add_argument('--dataset-name', default='random')
    job_args.add_argument('--no-cuda', action='store_true', default=False,
                          help='no cuda usage (default: %(default)s)')
    job_args.add_argument('--job', required=True,
                          choices=['train-dynamics',
                                   'evaluate-queries',
                                   'uncertainty-test',
                                   'uncertainty-test-plot'])
    # paths
    path_args = parser.add_argument_group('paths setup')
    path_args.add_argument('--d4rl-dataset-dir', type=Path,
                           default=Path(os.path.join('~/.d4rl', 'datasets')),
                           help="directory to store d4rl datasets (default: %(default)s)")
    path_args.add_argument('--policybazaar-dir', type=Path,
                           default=Path(os.path.join('~/.policybazaar')),
                           help="Directory to store policybazaar data (default: %(default)s)")
    path_args.add_argument('--result-dir', type=Path,
                           default=Path(os.path.join(os.getcwd(), 'results')),
                           help="Directory to store results (default: %(default)s)")

    # wandb setup
    wandb_args = parser.add_argument_group('wandb setup')
    wandb_args.add_argument('--wandb-project-name', default='cque-evaluations',
                            help='name of the Wandb project (default: %(default)s)')
    wandb_args.add_argument('--use-wandb', action='store_true',
                            help='Use Weight and bias visualization lib (default: %(default)s)')
    wandb_args.add_argument('--wandb-dir', default=os.path.join('~/.wandb'),
                            help="Directory Path to store wandb data (default: %(default)s)")

    # dynamics args
    dynamics_args = parser.add_argument_group('args for training dynamics')
    dynamics_args.add_argument('--dynamics-type', default='feed-forward',
                               choices=['feed-forward', 'autoregressive'],
                               help='type of dynamics model (default: %(default)s)')
    dynamics_args.add_argument('--deterministic', action='store_true', default=False,
                               help='if True, we use deterministic model otherwise '
                                    'stochastic (default: %(default)s)')
    dynamics_args.add_argument('--train-terminal', action='store_true', default=False,
                               help='enables training of terminal flag (default: %(default)s)')
    dynamics_args.add_argument('--use-dropout', action='store_true', default=False,
                               help='uses dropout within models (default: %(default)s)')
    dynamics_args.add_argument('--max-n-step-model', default=1, type=int,
                               help='n-step predictor for model (default: %(default)s)')

    dynamics_args.add_argument('--dynamics-seed', default=0, type=int,
                               help='seed for training dynamics (default: %(default)s)')
    dynamics_args.add_argument('--dynamics-test-interval', default=1, type=int,
                               help='epoch interval to test dynamics (default: %(default)s)')
    dynamics_args.add_argument('--dynamics-checkpoint-interval', type=int, default=1,
                               help='epoch interval to save dynamics '
                                    'checkpoint (default: %(default)s)')
    dynamics_args.add_argument('--hidden-size', type=int, default=200,
                               help='Hidden size for Linear Layers (default: %(default)s)')
    dynamics_args.add_argument('--dynamics-test-episodes', type=int, default=5,
                               help='No. of test episodes for evaluating '
                                    'dynamics with base policies (default: %(default)s)')
    dynamics_args.add_argument('--dynamics-test-episodes', type=int, default=5,
                               help='No. of test episodes for evaluating '
                                    'dynamics with base policies (default: %(default)s)')

    dynamics_args.add_argument('--num-epochs', type=int, default=100,
                               help='Epochs for training (default: %(default)s)')
    dynamics_args.add_argument('--batch-count', type=int, default=20,
                               help='Batches per epochs (default: %(default)s)')
    dynamics_args.add_argument('--dynamics-batch-size', type=int, default=256,
                               help='Batch size for Dynamics Learning (default: %(default)s)')
    dynamics_args.add_argument('--reward-loss-coeff', type=int, default=1,
                               help='Reward loss coefficient for training (default: %(default)s)')
    dynamics_args.add_argument('--observation-loss-coeff', type=int, default=1,
                               help='Obs. loss coefficient for training (default: %(default)s)')
    dynamics_args.add_argument('--terminal-loss-coeff', type=int, default=1,
                               help='Terminal loss coefficient for training (default: %(default)s)')
    dynamics_args.add_argument('--grad-clip-norm', type=float, default=5.0,
                               help='Gradient clipping norm (default: %(default)s)')
    dynamics_args.add_argument('--dynamics-lr', type=float, default=1e-3,
                               help='Learning rate for Dynamics (default: %(default)s)')

    dynamics_args.add_argument('--clip-obs', action='store_true', default=False,
                               help='clip the observation space with bounds (default: %(default)s)')
    dynamics_args.add_argument('--clip-reward', action='store_true', default=False,
                               help='clip the reward with dataset bounds (default: %(default)s)')
    dynamics_args.add_argument('--normalize-obs', action='store_true', default=False,
                               help='normalizes the observation space (default: %(default)s)')
    dynamics_args.add_argument('--normalize-reward', action='store_true', default=False,
                               help='normalizes the rewards (default: %(default)s)')
    dynamics_args.add_argument('--normalize-action', action='store_true', default=False,
                               help='normalizes the action space (default: %(default)s)')

    dynamics_args.add_argument('--add-observation-noise', action='store_true', default=False,
                               help='adds noise to observations  (default: %(default)s)')
    dynamics_args.add_argument('--add-reward-noise', action='store_true', default=False,
                               help='adds noise to rewards  (default: %(default)s)')
    dynamics_args.add_argument('--observation-noise-std', type=float, default=0.01,
                               help='std for adding noise to observation (default: %(default)s)')
    dynamics_args.add_argument('--reward-noise-std', type=float, default=0.01,
                               help='std for adding noise to reward (default: %(default)s)')

    dynamics_args.add_argument('--num-ensemble', default=1, type=int,
                               help='number of dynamics for ensemble (default: %(default)s)')
    dynamics_args.add_argument('--constant-prior', action='store_true', default=False,
                               help='adds a constant prior to each model (default: %(default)s)')

    # queries evaluation args
    queries_args = parser.add_argument_group('hyper-parameters for evaluating queries')
    queries_args.add_argument('--restore-dynamics-from-wandb', action='store_true', default=False,
                              help='restore model from wandb run. (default: %(default)s)')
    queries_args.add_argument('--wandb-dynamics-run-id', type=str,
                              help='Wandb run id if restoring model (default: %(default)s)')
    queries_args.add_argument('--query-eval-ensemble-mixture', action='store_true', default=False,
                              help='if enabled, mixes ensemble models for evaluation'
                                   '  (default: %(default)s)')
    queries_args.add_argument('--query-eval-runs', type=int, default=1,
                              help='Run count for each query (default: %(default)s)')
    queries_args.add_argument('--query-eval-step-batch-size', type=int, default=128,
                              help='batch size for query evaluation (default: %(default)s)')

    # uncertainty-test arguments
    uncertainty_test_args = parser.add_argument_group('args for uncertainty-test')
    uncertainty_test_args.add_argument('--uncertainty-test-type', default='ensemble-voting',
                                       choices=['paired-confidence-interval',
                                                'unpaired-confidence-interval',
                                                'ensemble-voting'])
    uncertainty_test_args.add_argument('--wandb-query-data-run-id', type=str,
                                       help='Wandb Query run id (default: %(default)s)')

    # Process arguments
    args = parser.parse_args(arg_str.split(" ") if arg_str else None)
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'

    return args, job_args, path_args, uncertainty_test_args, wandb_args, dynamics_args, queries_args


if __name__ == '__main__':
    args, job_args, path_args, uncertainty_test_args, \
        wandb_args, dynamics_args, queries_args = get_args()

    # d4rl setup
    os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
    os.environ['D4RL_DATASET_DIR'] = str(args.d4rl_dataset_dir)
    os.environ['POLICYBAZAAR_DIR'] = str(args.policybazaar_dir)

    if args.job == 'train-dynamics':
        config = BaseConfig(args, dynamics_args)
        if args.use_wandb:
            wandb.init(job_type=args.job,
                       dir=args.wandb_dir,
                       project=args.wandb_project_name + '-' + args.job,
                       settings=wandb.Settings(start_method="thread"))
            wandb.config.update(job_args)
            wandb.config.update(dynamics_args)
        train_dynamics(config)
    else:
        raise NotImplementedError()

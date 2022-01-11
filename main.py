import argparse
import json
import os
import random
from pathlib import Path

import cque
import numpy as np
import torch
import pandas as pd
import wandb
from core.config import BaseConfig
from core.train import train_dynamics
from core.utils import evaluate_queries
from wandb.plot import scatter


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
    dynamics_args.add_argument('--dataset-name', default='random',
                               help='name of the dataset')
    dynamics_args.add_argument('--dynamics-type', default='feed-forward',
                               choices=['feed-forward', 'autoregressive'],
                               help='type of dynamics model')
    dynamics_args.add_argument('--deterministic', action='store_true',
                               help='if True, we use deterministic model '
                                    'otherwise stochastic')
    dynamics_args.add_argument('--use-dropout', action='store_true',
                               help='uses dropout within models')
    dynamics_args.add_argument('--n-step-model', default=1, type=int,
                               help='n-step predictor for model ')
    dynamics_args.add_argument('--dynamics-seed', default=0, type=int,
                               help='seed for training dynamics ')
    dynamics_args.add_argument('--log-interval', default=1, type=int,
                               help='log interval for training dynamics')
    dynamics_args.add_argument('--dynamics-checkpoint-interval', type=int,
                               default=1, help='update interval to save'
                                               ' dynamics checkpoint ')
    dynamics_args.add_argument('--hidden-size', type=int, default=200,
                               help='hidden size for Linear Layers ')

    dynamics_args.add_argument('--update-count', type=int, default=100,
                               help='epochs for training ')
    dynamics_args.add_argument('--dynamics-batch-size', type=int, default=256,
                               help='batch size for Dynamics Learning ')
    dynamics_args.add_argument('--reward-loss-coeff', type=int, default=1,
                               help='reward loss coefficient for training ')
    dynamics_args.add_argument('--observation-loss-coeff', type=int, default=1,
                               help='obs. loss coefficient for training ')
    dynamics_args.add_argument('--grad-clip-norm', type=float, default=5.0,
                               help='gradient clipping norm')
    dynamics_args.add_argument('--dynamics-lr', type=float, default=1e-3,
                               help='learning rate for Dynamics')

    dynamics_args.add_argument('--normalize-obs', action='store_true',
                               help='normalizes the observation space ')
    dynamics_args.add_argument('--normalize-reward', action='store_true',
                               help='normalizes the rewards')
    dynamics_args.add_argument('--normalize-action', action='store_true',
                               help='normalizes the action space ')
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
    queries_args.add_argument('--ensemble-mixture',
                              action='store_true',
                              help='if enabled, mixes ensemble models '
                                   'for query evaluation  ')
    queries_args.add_argument('--eval-runs', type=int, default=1,
                              help='run count for each query evaluation')
    queries_args.add_argument('--eval-batch-size', type=int,
                              default=128,
                              help='batch size for query evaluation')
    queries_args.add_argument('--clip-obs', action='store_true',
                              help='clip the observation space with bounds '
                                   'for query evaluation')
    queries_args.add_argument('--clip-reward', action='store_true',
                              help='clip the reward with dataset bounds'
                                   ' for query evaluation')

    # uncertainty-test arguments
    uncertain_args = parser.add_argument_group('args for  uncertainty-test')
    uncertain_args.add_argument('--uncertainty-test-type',
                                default='ensemble-voting',
                                choices=['paired-confidence-interval',
                                         'unpaired-confidence-interval',
                                         'ensemble-voting'],
                                help='type of uncertainty test')
    uncertain_args.add_argument('--restore-query-eval-data-from-wandb',
                                action='store_true',
                                help='restore query evaluation data from wandb')
    uncertain_args.add_argument('--wandb-query-eval-data-run-id', type=str,
                                help='wandb run id  having query eval data')

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
        # set-up config
        if args.restore_dynamics_from_wandb:
            # get remote config from wandb
            assert args.wandb_dynamics_run_id is not None, \
                'w&b id cannot be {}'.format(args.wandb_dynamics_run_id)
            remote_config = wandb.Api().run(args.wandb_dynamics_run_id).config
            assert args.env_name == remote_config['env_name']
            for key in remote_config:
                if key in dynamics_args:
                    setattr(args, key, remote_config[key])
                    setattr(dynamics_args, key, remote_config[key])

            # download dynamics
            config = BaseConfig(args, dynamics_args)
            root, name = config.checkpoint_path
            os.makedirs(root, exist_ok=True)
            wandb.restore(name=name, run_path=args.wandb_dynamics_run_id,
                          replace=True, root=root)
        else:
            config = BaseConfig(args, dynamics_args)

        # setup experiment tracking
        if args.use_wandb:
            wandb.init(job_type=args.job,
                       dir=args.wandb_dir,
                       project=args.wandb_project_name + '-' + args.job,
                       settings=wandb.Settings(start_method="thread"))
            wandb.config.update({x.dest: vars(args)[x.dest]
                                 for x in job_args._group_actions})
            wandb.config.update({x.dest: vars(args)[x.dest]
                                 for x in dynamics_args._group_actions})
            wandb.config.update({x.dest: vars(args)[x.dest]
                                 for x in queries_args._group_actions})

        # dynamics setup
        assert os.path.exists(config.checkpoint_path), \
            'dynamics network not found: {}'.format(config.checkpoint_path)
        network = config.get_uniform_dynamics_network()
        state_dict = torch.load(config.checkpoint_path, torch.device('cpu'))
        print('state check-point update:{}'.format(state_dict['update_i']))
        network.load_state_dict(state_dict['network'])

        # set clipping flags
        if config.args.clip_obs:
            network.enable_obs_clip()
        if config.args.clip_reward:
            network.enable_reward_clip()

        network.eval()
        network = network.to(config.args.device)

        # query-evaluation
        queries = cque.get_queries(args.env_name)
        predicted_df = evaluate_queries(queries=queries,
                                        network=network,
                                        runs=args.eval_runs,
                                        batch_size=args.eval_batch_size,
                                        device=args.device,
                                        ensemble_mixture=args.ensemble_mixture)

        # store results:
        predicted_df.to_pickle(config.evaluate_queries_path)
        if args.use_wandb:
            wandb.run.summary["model-check-point"] = state_dict['epoch_i']

            table = wandb.Table(dataframe=predicted_df)
            wandb.log({'query-eval-data': table})
            wandb.log({"mean/q-value-comparison-a":
                           scatter(table, x="pred_a_mean", y="target_a",
                                   title="q-value-comparison-a")})
            wandb.log({"mean/q-value-comparison-b":
                           scatter(table, x="pred_b_mean", y="target_b",
                                   title="q-value-comparison-b")})
            wandb.log({"iqm/q-value-comparison-a":
                           scatter(table, x="pred_a_iqm", y="target_a",
                                   title="q-value-comparison-a")})
            wandb.log({"iqm/q-value-comparison-b":
                           scatter(table, x="pred_b_iqm", y="target_b",
                                   title="q-value-comparison-b")})
            wandb.log({"median/q-value-comparison-a":
                           scatter(table, x="pred_a_median", y="target_a",
                                   title="q-value-comparison-a")})
            wandb.log({"median/q-value-comparison-b":
                           scatter(table, x="pred_b_median", y="target_b",
                                   title="q-value-comparison-b")})

    elif args.job == 'uncertainty-test':
        # setup config and restore query evaluation data
        if args.restore_query_eval_data_from_wandb:
            assert args.wandb_query_eval_data_run_id is not None, \
                'w&b id cannot be {}'.format(args.wandb_query_data_run_id)
            run = wandb.Api().run(args.wandb_query_eval_data_run_id)
            table_file_path = run.summary.get('query-eval-data').get("path")
            table_file = wandb.restore(table_file_path,
                                       run_path=args.wandb_query_data_run_id)
            table_str = table_file.read()
            table_dict = json.loads(table_str)
            query_eval_df = pd.DataFrame(**table_dict)
        else:
            config = BaseConfig(args, dynamics_args)
            query_eval_df = pd.read_pickle(config.evaluate_queries_path)

        # # get targets and estimates
        # return_a_ensemble = np.concatenate([np.expand_dims(query_df['pred_a_{}'.format(ensemble_i)].values, 1)
        #                                     for ensemble_i in range(args.num_ensemble)], axis=1)
        # return_b_ensemble = np.concatenate([np.expand_dims(query_df['pred_b_{}'.format(ensemble_i)].values, 1)
        #                                     for ensemble_i in range(args.num_ensemble)], axis=1)
        #
        # horizon = query_eval_df['horizon'].values
        # query_horizon_candidates = np.unique(query_horizon, axis=0)
        # target = query_eval_df['target'].values
        #
        # ensemble_voting(query_eval_df)
    else:
        raise NotImplementedError()

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
from core.uncertainty import ensemble_voting as ev, confidence_interval as ci
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
    wandb_args.add_argument('--wandb-dir', default=os.path.join('~/'),
                            help="directory Path to store wandb data")

    # dynamics args
    dynamics_args = parser.add_argument_group('args for training dynamics')
    job_args.add_argument('--env-name', default='HalfCheetah-v2',
                          help='name of the environment')
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
    queries_args.add_argument('--ensemble-mixture', action='store_true',
                              help='if enabled, randomly select a models at'
                                   ' each step of query evaluation  ')
    queries_args.add_argument('--eval-runs', type=int, default=1,
                              help='run count for each query evaluation')
    queries_args.add_argument('--reset-n-step', type=int, default=1,
                              help='run count for each query evaluation')
    queries_args.add_argument('--eval-batch-size', type=int, default=128,
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
    os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = "1"
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
                'wandb-dynamics-run-id cannot be None'
            remote_config = wandb.Api().run(args.wandb_dynamics_run_id).config
            assert args.env_name == remote_config['env_name']

            # preserve original dynamics args
            for _arg in dynamics_args._group_actions:
                setattr(args, _arg.dest, remote_config[_arg.dest])
                setattr(dynamics_args, _arg.dest, remote_config[_arg.dest])

            # download dynamics
            config = BaseConfig(args, dynamics_args)
            root = os.path.dirname(config.checkpoint_path)
            name = os.path.basename(config.checkpoint_path)
            os.makedirs(root, exist_ok=True)
            wandb.restore(name=name, run_path=args.wandb_dynamics_run_id,
                          replace=True, root=root)
        else:
            config = BaseConfig(args, dynamics_args)

        assert args.reset_n_step <= config.args.n_step_model

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
        print('state check-point update:{}'.format(state_dict['update']))
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
                                        ensemble_mixture=args.ensemble_mixture,
                                        reset_n_step=args.reset_n_step)
        predicted_df.to_pickle(config.evaluate_queries_path(args,
                                                            queries_args))

        # log data on wandb
        if args.use_wandb:
            wandb.run.summary["model-check-point"] = state_dict['update']

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
        query_eval_df = None
        # restore query evaluation data
        if args.restore_query_eval_data_from_wandb:
            assert args.wandb_query_eval_data_run_id is not None, \
                'wandb-query-eval-data-run-id cannot be None'
            run = wandb.Api().run(args.wandb_query_eval_data_run_id)
            remote_config = run.config
            assert args.env_name == remote_config['env_name']

            # preserve original dynamics args
            for _arg in dynamics_args._group_actions:
                setattr(args, _arg.dest, remote_config[_arg.dest])
                setattr(dynamics_args, _arg.dest, remote_config[_arg.dest])

            # preserve original query eval args
            for _arg in queries_args._group_actions:
                setattr(args, _arg.dest, remote_config[_arg.dest])
                setattr(queries_args, _arg.dest, remote_config[_arg.dest])

            # download query-evaluation data
            table_file_path = run.summary.get('query-eval-data').get("path")
            table_file = wandb.restore(table_file_path,
                                       run_path=args.wandb_query_eval_data_run_id)
            table_str = table_file.read()
            table_dict = json.loads(table_str)
            query_eval_df = pd.DataFrame(**table_dict)

        config = BaseConfig(args, dynamics_args)
        if query_eval_df is None:
            query_eval_df = pd.read_pickle(
                config.evaluate_queries_path(args, queries_args))

        # ################
        # uncertainty-test
        # ################

        if config.args.uncertainty_test == 'ensemble-voting':
            ensemble_df, horizon_df = ev(query_eval_df,
                                         ensemble_size_interval=10,
                                         num_ensemble=config.args.num_ensemble,
                                         confidence_interval=0.1)
        elif config.args.uncertainty_test == 'paired-confidence-interval':
            ensemble_df, horizon_df = ci(query_eval_df,
                                         ensemble_size_interval=10,
                                         num_ensemble=config.args.num_ensemble,
                                         step=0.1,
                                         paired=True)
        elif config.args.uncertainty_test == 'unpaired-confidence-interval':
            ensemble_df, horizon_df = ci(query_eval_df,
                                         ensemble_size_interval=10,
                                         num_ensemble=config.args.num_ensemble,
                                         step=0.1,
                                         paired=False)
        else:
            raise NotImplementedError(
                '{} is not implemented'.format(config.args.uncetainty_test))

        # save-data
        # setup for saving data on wandb
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
            wandb.config.update({x.dest: vars(args)[x.dest]
                                 for x in uncertainty_args._group_actions})

            ensemble_df_table = wandb.Table(dataframe=ensemble_df)
            horizon_df_table = wandb.Table(dataframe=horizon_df)
            wandb.log({'ensemble-data': ensemble_df,
                       'horizon-data': horizon_df})

            for category, _category_df in [('ensemble_count', ensemble_df),
                                           ('horizon', horizon_df)]:
                categories = _category_df[category].unique()
                conf_threshold = _category_df['confidence_threshold'].unique()
                categories.sort()
                conf_threshold.sort()
                ys = {'accuracy': [], 'abstain': [], 'abstain_count': []}
                for cat in categories:
                    _filter = _category_df[category] == cat
                    _sub_df = _category_df[_filter].sort_values(by=['confidence_threshold'])
                    ys['accuracy'].append(_sub_df['accuracy'].values.tolist())
                    ys['abstain'].append(_sub_df['abstain'].values.tolist())
                    ys['abstain_count'].append(_sub_df['abstain_count'].values.tolist())

                for key, val in ys.items():
                    _plot = wandb.plot.line_series(
                        xs=conf_threshold,
                        ys=val,
                        keys=["{}:{}".format(category, e) for e in categories],
                        title='{}-{}'.format(category, key),
                        xname="confidence-threshold")
                    wandb.log({'{}-{}'.format(category, key): _plot})

    else:
        raise NotImplementedError()

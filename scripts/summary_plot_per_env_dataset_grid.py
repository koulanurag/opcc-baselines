import wandb
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def row_plot(row_data, row_axs, horizons, conf_attr_name, dataset_name):
    for horizon_i, horizon in enumerate(horizons):
        cell_data = row_data[(row_data['dataset_name'] == dataset_name)
                             & (row_data['horizon'] == horizon)]

        conf_thresholds = sorted(np.unique(cell_data[conf_attr_name].values))
        accuracy_mean_data = []
        abstain_mean_data = []
        accuracy_std_data = []
        abstain_std_data = []

        for conf_threshold in conf_thresholds:
            conf_data = cell_data[(cell_data[conf_attr_name] == conf_threshold)]
            accuracy_mean = np.mean(conf_data['accuracy'].values)
            abstain_mean = np.mean(conf_data['abstain'].values)
            accuracy_mean_data.append(accuracy_mean)
            abstain_mean_data.append(abstain_mean)

            accuracy_std = np.std(conf_data['accuracy'].values)
            abstain_std = np.std(conf_data['abstain'].values)

            accuracy_std_data.append(accuracy_std)
            abstain_std_data.append(abstain_std)
        assert len(accuracy_mean_data) == len(conf_thresholds)
        row_axs[horizon_i].plot(abstain_mean_data, accuracy_mean_data)


def main():
    # combine data from multiple runs into a single dataframe
    api = wandb.Api()
    ensemble_data_df = []
    horizon_data_df = []
    run_count = 0
    for run in api.runs('koulanurag/cque-baselines-final-uncertainty-test',
                        filters={'state': 'finished', 'config.env_name': 'Walker2d-v2'}):
        run_count += 1
        print(run_count, run)
        table_file = wandb.restore(run.summary.get('ensemble-data').get("path"),
                                   run_path='/'.join(run.path))
        table_str = table_file.read()
        table_dict = json.loads(table_str)
        query_df = pd.DataFrame(**table_dict)

        query_df['env_name'] = [run.config['env_name'] for _ in range(len(query_df))]
        query_df['dataset_name'] = [run.config['dataset_name'] for _ in range(len(query_df))]
        query_df['dynamics_type'] = [run.config['dynamics_type'] for _ in range(len(query_df))]
        query_df['uncertainty_test_type'] = [run.config['uncertainty_test_type'] for _ in range(len(query_df))]
        query_df['deterministic'] = [run.config['deterministic'] for _ in range(len(query_df))]
        query_df['constant_prior'] = [run.config['constant_prior'] for _ in range(len(query_df))]
        query_df['ensemble_mixture'] = [run.config['ensemble_mixture'] for _ in range(len(query_df))]
        ensemble_data_df.append(query_df)

        table_file = wandb.restore(run.summary.get('horizon-data').get("path"),
                                   run_path='/'.join(run.path))
        table_str = table_file.read()
        table_dict = json.loads(table_str)
        query_df = pd.DataFrame(**table_dict)

        query_df['env_name'] = [run.config['env_name']
                                for _ in range(len(query_df))]
        query_df['dataset_name'] = [run.config['dataset_name']
                                    for _ in range(len(query_df))]
        query_df['dynamics_type'] = [run.config['dynamics_type']
                                     for _ in range(len(query_df))]
        query_df['uncertainty_test_type'] = [run.config['uncertainty_test_type']
                                             for _ in range(len(query_df))]
        query_df['deterministic'] = ['deterministic' if run.config['deterministic'] else 'stochastic'
                                     for _ in range(len(query_df))]
        query_df['constant_prior'] = ['constant prior' if run.config['constant_prior'] else 'no constant prior'
                                      for _ in range(len(query_df))]
        query_df['ensemble_mixture'] = ['mixture' if run.config['ensemble_mixture'] else 'no mixture'
                                        for _ in range(len(query_df))]
        horizon_data_df.append(query_df)

    ensemble_data_df = pd.concat(ensemble_data_df)
    horizon_data_df = pd.concat(horizon_data_df)
    uncertainty_test_types = np.unique(ensemble_data_df['uncertainty_test_type'].values)
    envs = np.unique(ensemble_data_df['env_name'].values)

    # overall fig
    for uncertainty_test_type in ['ensemble-voting', 'unpaired-confidence-interval', 'paired-confidence-interval']:
        for env in envs.tolist():
            dataset_names = np.unique(
                ensemble_data_df[(ensemble_data_df['uncertainty_test_type'] == uncertainty_test_type)
                                 & (ensemble_data_df['env_name'] == env)]['dataset_name'].values)
            ensemble_counts = np.unique(
                ensemble_data_df[(ensemble_data_df['uncertainty_test_type'] == uncertainty_test_type)
                                 & (ensemble_data_df['env_name'] == env)]['ensemble_count'].values)
            max_ensemble_count = max(ensemble_counts)

            horizons = np.unique(horizon_data_df[
                                     (horizon_data_df['uncertainty_test_type'] == uncertainty_test_type)
                                     & (horizon_data_df['env_name'] == env)]['horizon'].values)

            # & (horizon_data_df['dataset_name'] == dataset_name)
            horizons = sorted(horizons)
            num_rows = 5
            num_cols = len(horizons)
            col_names = ['horizon:{}'.format(h) for h in horizons]
            fig, axs = plt.subplots(num_rows, num_cols,
                                    figsize=(2.2 * num_cols, 2 * num_rows),
                                    layout='constrained', squeeze=False)

            for dataset_name in dataset_names:

                if uncertainty_test_type == 'ensemble-voting':
                    conf_attr_name = 'confidence_threshold'
                else:
                    conf_attr_name = 'confidence_level'

                base_data_filter = (horizon_data_df['env_name'] == env) \
                                   & (horizon_data_df['dataset_name'] == dataset_name) \
                                   & (horizon_data_df['uncertainty_test_type'] == uncertainty_test_type)

                # base-model row
                base_prior = 'constant prior'
                base_deter = 'stochastic'
                base_mixture = 'no mixture'
                base_dyn_type = 'feed-forward'
                fig_title = "base-model: " + "|".join([base_prior, base_deter,
                                                       base_mixture,
                                                       base_dyn_type])

                # #####################################################
                # labels
                for col_i, col_title in enumerate(col_names):
                    if col_title is not None:
                        axs[0, col_i].set_title(col_title)
                    axs[num_rows - 1, col_i].set(xlabel='abstain')

                for row_i in range(num_rows):
                    for _j in range(num_cols):
                        axs[row_i, _j].grid(True)
                    axs[row_i, 0].set(ylabel='accuracy')
                # #####################################################

                # #####################################################
                row_titles = ['base-model']
                row_data = horizon_data_df[base_data_filter
                                           & (horizon_data_df['constant_prior'] == base_prior)
                                           & (horizon_data_df['deterministic'] == base_deter)
                                           & (horizon_data_df['ensemble_mixture'] == base_mixture)
                                           & (horizon_data_df['dynamics_type'] == base_dyn_type)]
                row_plot(row_data, axs[len(row_titles) - 1], horizons, conf_attr_name, dataset_name)

                # prior
                _filter_name = 'no constant prior'
                row_titles.append(_filter_name)
                row_data = horizon_data_df[base_data_filter
                                           & (horizon_data_df['constant_prior'] == _filter_name)
                                           & (horizon_data_df['deterministic'] == base_deter)
                                           & (horizon_data_df['ensemble_mixture'] == base_mixture)
                                           & (horizon_data_df['dynamics_type'] == base_dyn_type)]
                row_plot(row_data, axs[len(row_titles) - 1], horizons, conf_attr_name, dataset_name)

                # deterministic
                _filter_name = 'deterministic'
                row_titles.append(_filter_name)
                row_data = horizon_data_df[base_data_filter
                                           & (horizon_data_df['constant_prior'] == base_prior)
                                           & (horizon_data_df['deterministic'] == _filter_name)
                                           & (horizon_data_df['ensemble_mixture'] == base_mixture)
                                           & (horizon_data_df['dynamics_type'] == base_dyn_type)]
                row_plot(row_data, axs[len(row_titles) - 1], horizons, conf_attr_name, dataset_name)

                # ensemble-mixture
                _filter_name = 'mixture'
                row_titles.append(_filter_name)
                row_data = horizon_data_df[base_data_filter
                                           & (horizon_data_df['constant_prior'] == base_prior)
                                           & (horizon_data_df['deterministic'] == base_deter)
                                           & (horizon_data_df['ensemble_mixture'] == _filter_name)
                                           & (horizon_data_df['dynamics_type'] == base_dyn_type)]
                row_plot(row_data, axs[len(row_titles) - 1], horizons, conf_attr_name, dataset_name)

                # dynamics-type
                _filter_name = 'autoregressive'
                row_titles.append(_filter_name)
                row_data = horizon_data_df[base_data_filter
                                           & (horizon_data_df['constant_prior'] == base_prior)
                                           & (horizon_data_df['deterministic'] == base_deter)
                                           & (horizon_data_df['ensemble_mixture'] == base_mixture)
                                           & (horizon_data_df['dynamics_type'] == _filter_name)]
                row_plot(row_data, axs[len(row_titles) - 1], horizons, conf_attr_name, dataset_name)

            for row_i, row_title in enumerate(row_titles):
                axs[row_i, num_cols - 1].text(1.1, 0.5, row_title.upper(), style='italic',
                                              horizontalalignment='center', verticalalignment='center',
                                              transform=axs[row_i, num_cols - 1].transAxes,
                                              rotation=270,
                                              bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})

            fig.suptitle(fig_title)
            _path = os.path.join(os.getcwd(), 'uncertainty_test_plots', env, uncertainty_test_type)
            os.makedirs(_path, exist_ok=True)
            fig.savefig(os.path.join(_path, 'accuracy_vs_abstain.png'), bbox_inches="tight")
            plt.close(fig)

                # ###################################
                # base_data_filter = (ensemble_data_df['env_name'] == env) \
                #                    & (ensemble_data_df['dataset_name'] == dataset_name) \
                #                    & (ensemble_data_df['uncertainty_test_type'] == uncertainty_test_type)
                # fig, axs = plt.subplots(1, 1,
                #                         figsize=(2.2 * num_cols, 2 * num_rows),
                #                         layout='constrained', squeeze=False)
                # for ensemble_count in ensemble_counts:
                #     cell_data = ensemble_data_df[base_data_filter &
                #                                  (ensemble_data_df['ensemble_count'] == ensemble_count)]
                #
                #     conf_thresholds = sorted(np.unique(cell_data[conf_attr_name].values))
                #     accuracy_mean_data = []
                #     abstain_mean_data = []
                #     accuracy_std_data = []
                #     abstain_std_data = []
                #
                #     for conf_threshold in conf_thresholds:
                #         conf_data = ensemble_data_df[base_data_filter
                #                                      & (ensemble_data_df['ensemble_count'] == ensemble_count)
                #                                      & (ensemble_data_df[conf_attr_name] == conf_threshold)]
                #         accuracy_mean = np.mean(conf_data['accuracy'].values)
                #         abstain_mean = np.mean(conf_data['abstain'].values)
                #         accuracy_mean_data.append(accuracy_mean)
                #         abstain_mean_data.append(abstain_mean)
                #
                #         accuracy_std = np.std(conf_data['accuracy'].values)
                #         abstain_std = np.std(conf_data['abstain'].values)
                #
                #         accuracy_std_data.append(accuracy_std)
                #         abstain_std_data.append(abstain_std)
                #     assert len(accuracy_mean_data) == len(conf_thresholds)
                #     axs[0, 0].plot(abstain_mean_data, accuracy_mean_data,
                #                    label=ensemble_count)
                #
                # axs[0, 0].legend()
                # fig.savefig(os.path.join(_path, 'ensemble_count.png'),
                #             bbox_inches="tight")
                # plt.close(fig)


if __name__ == '__main__':
    main()

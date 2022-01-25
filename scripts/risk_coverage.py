import wandb
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


def row_plot(row_data, rpp_row_data, row_axs, col_key_values, conf_attr_name, dataset_name, col_key, value_risk=False):
    for _i, val in enumerate(col_key_values):
        cell_data = row_data[(row_data['dataset_name'] == dataset_name)
                             & (row_data[col_key] == val)]

        conf_thresholds = sorted(np.unique(cell_data[conf_attr_name].values))
        accuracy_mean_data = []
        abstain_mean_data = []
        accuracy_std_data = []
        abstain_std_data = []
        value_regret_risk_mean_data = []
        value_regret_risk_std_data = []

        coverage_mean_data = []
        coverage_std_data = []

        for conf_threshold in conf_thresholds:
            conf_data = cell_data[(cell_data[conf_attr_name] == conf_threshold)]
            accuracy_mean = np.mean(conf_data['accuracy'].values)
            abstain_mean = np.mean(conf_data['abstain'].values)
            accuracy_mean_data.append(accuracy_mean)
            abstain_mean_data.append(abstain_mean)
            value_regret_risk_mean_data.append(np.mean(conf_data['value_regret_risk'].values))

            accuracy_std = np.std(conf_data['accuracy'].values)
            abstain_std = np.std(conf_data['abstain'].values)
            accuracy_std_data.append(accuracy_std)
            abstain_std_data.append(abstain_std)
            value_regret_risk_std_data.append(np.std(conf_data['value_regret_risk'].values))

            coverage_data = (conf_data.non_abstain_count.values /
                             (conf_data.abstain_count.values +
                              conf_data.non_abstain_count.values))

            coverage_mean_data.append(np.mean(coverage_data))
            coverage_std_data.append(np.std(coverage_data))

        assert len(accuracy_mean_data) == len(conf_thresholds)

        if not value_risk:
            risk = (1 - np.array(accuracy_mean_data)).tolist()
        else:
            risk = value_regret_risk_mean_data

        try:
            coverage = coverage_mean_data
            c_r_data = np.array(sorted([(coverage[i], risk[i]) for i in range(len(risk))]))
            risk = c_r_data[:, 1].tolist()
            coverage = c_r_data[:, 0].tolist()

            risk = [0] + risk
            coverage = [0] + coverage

            auc = metrics.auc(coverage, risk)
            row_axs[_i].plot(coverage, risk)

            coverage_interval = 0.05
            bins = [x for x in np.arange(0, 1.0001, coverage_interval)]
            coverage_resolution_score = np.unique(np.digitize(coverage, bins)).size / len(bins)
            row_axs[_i].text(0.5, 0.5, 'auc:{:.2f} \n coverage score:{:.2f} \n rpp : {:.2f}'.format(auc,
                                                                                                    coverage_resolution_score,
                                                                                                    rpp_row_data[(
                                                                                                                             rpp_row_data[
                                                                                                                                 'dataset_name'] == dataset_name) & (
                                                                                                                             rpp_row_data[
                                                                                                                                 col_key] == val)][
                                                                                                        'rpp'].values.mean()),
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=row_axs[_i].transAxes)
        except:
            print('no data')
            pass


def plot(data_df, rpp_df, dataset_names, fig, axs, num_cols, env,
         uncertainty_test_type, col_names, num_rows, col_key_values,
         conf_attr_name, col_key, plot_name, value_risk):
    for dataset_name in dataset_names:
        base_data_filter = (data_df['env_name'] == env) \
                           & (data_df['dataset_name'] == dataset_name) \
                           & (data_df['uncertainty_test_type'] == uncertainty_test_type)

        base_rpp_data_filter = (rpp_df['env_name'] == env) \
                               & (rpp_df['dataset_name'] == dataset_name) \
                               & (rpp_df['uncertainty_test_type'] == uncertainty_test_type)

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
        # #####################################################
        for col_i, col_title in enumerate(col_names):
            if col_title is not None:
                axs[0, col_i].set_title(col_title)
            axs[num_rows - 1, col_i].set(xlabel='coverage')

        for row_i in range(num_rows):
            for _j in range(num_cols):
                axs[row_i, _j].grid(True)
            axs[row_i, 0].set(ylabel='risk')
        # #####################################################
        # #####################################################

        row_titles = ['base-model']
        row_data = data_df[base_data_filter
                           & (data_df['constant_prior'] == base_prior)
                           & (data_df['deterministic'] == base_deter)
                           & (data_df['ensemble_mixture'] == base_mixture)
                           & (data_df['dynamics_type'] == base_dyn_type)]
        row_rpp_data = rpp_df[base_rpp_data_filter
                              & (rpp_df['constant_prior'] == base_prior)
                              & (rpp_df['deterministic'] == base_deter)
                              & (rpp_df['ensemble_mixture'] == base_mixture)
                              & (rpp_df['dynamics_type'] == base_dyn_type)]
        row_plot(row_data, row_rpp_data, axs[len(row_titles) - 1], col_key_values,
                 conf_attr_name, dataset_name, col_key, value_risk)

        # prior
        _filter_name = 'no constant prior'
        row_titles.append(_filter_name)
        row_data = data_df[base_data_filter
                           & (data_df['constant_prior'] == _filter_name)
                           & (data_df['deterministic'] == base_deter)
                           & (data_df['ensemble_mixture'] == base_mixture)
                           & (data_df['dynamics_type'] == base_dyn_type)]
        row_rpp_data = rpp_df[base_rpp_data_filter
                              & (rpp_df['constant_prior'] == _filter_name)
                              & (rpp_df['deterministic'] == base_deter)
                              & (rpp_df['ensemble_mixture'] == base_mixture)
                              & (rpp_df['dynamics_type'] == base_dyn_type)]
        row_plot(row_data, row_rpp_data, axs[len(row_titles) - 1], col_key_values, conf_attr_name, dataset_name,
                 col_key, value_risk)

        # deterministic
        _filter_name = 'deterministic'
        row_titles.append(_filter_name)
        row_data = data_df[base_data_filter
                           & (data_df['constant_prior'] == base_prior)
                           & (data_df['deterministic'] == _filter_name)
                           & (data_df['ensemble_mixture'] == base_mixture)
                           & (data_df['dynamics_type'] == base_dyn_type)]
        row_rpp_data = rpp_df[base_rpp_data_filter
                              & (rpp_df['constant_prior'] == base_prior)
                              & (rpp_df['deterministic'] == _filter_name)
                              & (rpp_df['ensemble_mixture'] == base_mixture)
                              & (rpp_df['dynamics_type'] == base_dyn_type)]
        row_plot(row_data, row_rpp_data, axs[len(row_titles) - 1], col_key_values, conf_attr_name, dataset_name,
                 col_key, value_risk)

        # ensemble-mixture
        _filter_name = 'mixture'
        row_titles.append(_filter_name)
        row_data = data_df[base_data_filter
                           & (data_df['constant_prior'] == base_prior)
                           & (data_df['deterministic'] == base_deter)
                           & (data_df['ensemble_mixture'] == _filter_name)
                           & (data_df['dynamics_type'] == base_dyn_type)]
        row_rpp_data = rpp_df[base_rpp_data_filter
                              & (rpp_df['constant_prior'] == base_prior)
                              & (rpp_df['deterministic'] == base_deter)
                              & (rpp_df['ensemble_mixture'] == _filter_name)
                              & (rpp_df['dynamics_type'] == base_dyn_type)]
        row_plot(row_data, row_rpp_data, axs[len(row_titles) - 1], col_key_values, conf_attr_name, dataset_name,
                 col_key, value_risk)

        # dynamics-type
        _filter_name = 'autoregressive'
        row_titles.append(_filter_name)
        row_data = data_df[base_data_filter
                           & (data_df['constant_prior'] == base_prior)
                           & (data_df['deterministic'] == base_deter)
                           & (data_df['ensemble_mixture'] == base_mixture)
                           & (data_df['dynamics_type'] == _filter_name)]
        row_rpp_data = rpp_df[base_rpp_data_filter
                              & (rpp_df['constant_prior'] == base_prior)
                              & (rpp_df['deterministic'] == base_deter)
                              & (rpp_df['ensemble_mixture'] == base_mixture)
                              & (rpp_df['dynamics_type'] == _filter_name)]
        row_plot(row_data, row_rpp_data, axs[len(row_titles) - 1], col_key_values, conf_attr_name, dataset_name,
                 col_key, value_risk)

        for row_i, row_title in enumerate(row_titles):
            axs[row_i, num_cols - 1].text(1.1, 0.5, row_title.upper(), style='italic',
                                          horizontalalignment='center', verticalalignment='center',
                                          transform=axs[row_i, num_cols - 1].transAxes,
                                          rotation=270,
                                          bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})

        fig.suptitle(fig_title)
        _path = os.path.join(os.getcwd(), 'uncertainty_test_plots', env, dataset_name, uncertainty_test_type)
        os.makedirs(_path, exist_ok=True)
        fig.savefig(os.path.join(_path, '{}.png'.format(plot_name)), bbox_inches="tight")
        [ax.clear() for ax in axs.flat]
    plt.close(fig)


def main():
    # combine data from multiple runs into a single dataframe
    api = wandb.Api()
    ensemble_data_df = []
    ensemble_rpp_df = []
    horizon_data_df = []
    horizon_rpp_df = []
    run_count = 0
    for run in api.runs('koulanurag/cque-baselines-final-uncertainty-test',
                        filters={'state': 'finished', 'config.env_name': 'd4rl:maze2d-open-v0',
                                 }):
        try:
            run_count += 1
            print(run_count, run)
            table_file = wandb.restore(run.summary.get('ensemble-data').get("path"),
                                       run_path='/'.join(run.path))
            table_str = table_file.read()
            table_dict = json.loads(table_str)
            query_df = pd.DataFrame(**table_dict)
            rpp_df = pd.DataFrame(**json.loads((wandb.restore(run.summary.get('ensemble-rpp-data').get("path"),
                                                              run_path='/'.join(run.path))).read()))

            for _df in [query_df, rpp_df]:
                _df['env_name'] = [run.config['env_name'] for _ in range(len(_df))]
                _df['dataset_name'] = [run.config['dataset_name'] for _ in range(len(_df))]
                _df['dynamics_type'] = [run.config['dynamics_type'] for _ in range(len(_df))]
                _df['uncertainty_test_type'] = [run.config['uncertainty_test_type'] for _ in range(len(_df))]
                _df['deterministic'] = ['deterministic' if run.config['deterministic'] else 'stochastic'
                                        for _ in range(len(_df))]
                _df['constant_prior'] = ['constant prior' if run.config['constant_prior'] else 'no constant prior'
                                         for _ in range(len(_df))]
                _df['ensemble_mixture'] = ['mixture' if run.config['ensemble_mixture'] else 'no mixture'
                                           for _ in range(len(_df))]
            ensemble_data_df.append(query_df)
            ensemble_rpp_df.append(rpp_df)

            table_file = wandb.restore(run.summary.get('horizon-data').get("path"),
                                       run_path='/'.join(run.path))
            table_str = table_file.read()
            table_dict = json.loads(table_str)
            query_df = pd.DataFrame(**table_dict)
            rpp_df = pd.DataFrame(**json.loads((wandb.restore(run.summary.get('horizon-rpp-data').get("path"),
                                                              run_path='/'.join(run.path))).read()))
            for _df in [query_df, rpp_df]:
                _df['env_name'] = [run.config['env_name']
                                   for _ in range(len(_df))]
                _df['dataset_name'] = [run.config['dataset_name']
                                       for _ in range(len(_df))]
                _df['dynamics_type'] = [run.config['dynamics_type']
                                        for _ in range(len(_df))]
                _df['uncertainty_test_type'] = [run.config['uncertainty_test_type']
                                                for _ in range(len(_df))]
                _df['deterministic'] = ['deterministic' if run.config['deterministic'] else 'stochastic'
                                        for _ in range(len(_df))]
                _df['constant_prior'] = ['constant prior' if run.config['constant_prior'] else 'no constant prior'
                                         for _ in range(len(_df))]
                _df['ensemble_mixture'] = ['mixture' if run.config['ensemble_mixture'] else 'no mixture'
                                           for _ in range(len(_df))]

            horizon_data_df.append(query_df)
            horizon_rpp_df.append(rpp_df)
        except:
            print("ISSUE:", run.path)

    ensemble_data_df = pd.concat(ensemble_data_df)
    horizon_data_df = pd.concat(horizon_data_df)

    ensemble_rpp_df = pd.concat(ensemble_rpp_df)
    horizon_rpp_df = pd.concat(horizon_rpp_df)

    uncertainty_test_types = np.unique(ensemble_data_df['uncertainty_test_type'].values)
    envs = np.unique(ensemble_data_df['env_name'].values)

    # overall fig
    for uncertainty_test_type in ['ensemble-voting', 'unpaired-confidence-interval', 'paired-confidence-interval']:
        for env in envs.tolist():
            dataset_names = np.unique(
                ensemble_data_df[(ensemble_data_df['uncertainty_test_type'] == uncertainty_test_type)
                                 & (ensemble_data_df['env_name'] == env)]['dataset_name'].values)
            ensemble_counts = sorted(np.unique(
                ensemble_data_df[(ensemble_data_df['uncertainty_test_type'] == uncertainty_test_type)
                                 & (ensemble_data_df['env_name'] == env)]['ensemble_count'].values))
            max_ensemble_count = max(ensemble_counts)

            horizons = np.unique(horizon_data_df[
                                     (horizon_data_df['uncertainty_test_type'] == uncertainty_test_type)
                                     & (horizon_data_df['env_name'] == env)]['horizon'].values)
            horizons = sorted(horizons)
            num_rows = 5
            horizon_col_names = ['horizon:{}'.format(h) for h in horizons]
            horizon_value_risk_fig, horizon_value_risk_axs = plt.subplots(num_rows, len(horizon_col_names),
                                                                          figsize=(
                                                                          2.2 * len(horizon_col_names), 2 * num_rows),
                                                                          layout='constrained', squeeze=False)
            horizon_risk_fig, horizon_risk_axs = plt.subplots(num_rows, len(horizon_col_names),
                                                              figsize=(2.2 * len(horizon_col_names), 2 * num_rows),
                                                              layout='constrained', squeeze=False)

            if uncertainty_test_type == 'ensemble-voting':
                conf_attr_name = 'confidence_threshold'
            else:
                conf_attr_name = 'confidence_level'

            plot(horizon_data_df, horizon_rpp_df, dataset_names, horizon_value_risk_fig, horizon_value_risk_axs,
                 len(horizons), env,
                 uncertainty_test_type, horizon_col_names, num_rows, horizons,
                 conf_attr_name, 'horizon', 'horizon_value_regret_risk_vs_coverage', value_risk=True)
            plot(horizon_data_df, horizon_rpp_df, dataset_names, horizon_risk_fig, horizon_risk_axs, len(horizons), env,
                 uncertainty_test_type, horizon_col_names, num_rows, horizons,
                 conf_attr_name, 'horizon', 'horizon_risk_vs_coverage', value_risk=False)

            ensemble_col_names = ['ensemble:{}'.format(h) for h in ensemble_counts]
            ensemble_value_risk_fig, ensemble_value_risk_axs = plt.subplots(num_rows, len(ensemble_col_names),
                                                                            figsize=(2.2 * len(ensemble_col_names),
                                                                                     2 * num_rows),
                                                                            layout='constrained', squeeze=False)
            ensemble_fig, ensemble_count_axs = plt.subplots(num_rows, len(ensemble_col_names),
                                                            figsize=(2.2 * len(ensemble_col_names), 2 * num_rows),
                                                            layout='constrained', squeeze=False)
            plot(ensemble_data_df, ensemble_rpp_df, dataset_names, ensemble_value_risk_fig, ensemble_value_risk_axs,
                 len(ensemble_counts), env,
                 uncertainty_test_type, ensemble_col_names, num_rows, ensemble_counts,
                 conf_attr_name, 'ensemble_count', 'ensemble_count_value_regret_risk_vs_coverage', value_risk=True)
            plot(ensemble_data_df, ensemble_rpp_df, dataset_names, ensemble_fig, ensemble_count_axs,
                 len(ensemble_counts), env,
                 uncertainty_test_type, ensemble_col_names, num_rows, ensemble_counts,
                 conf_attr_name, 'ensemble_count', 'ensemble_count_risk_vs_coverage', value_risk=False)


if __name__ == '__main__':
    main()

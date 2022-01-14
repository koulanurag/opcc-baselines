import wandb
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # combine data from multiple runs into a single dataframe
    api = wandb.Api()
    ensemble_data_df = []
    horizon_data_df = []
    for run in api.runs('koulanurag/cque-baselines-1-uncertainty-test',
                        filters={'state': 'finished'}):
        print(run)
        table_file = wandb.restore(run.summary.get('ensemble-data').get("path"),
                                   run_path='/'.join(run.path))
        table_str = table_file.read()
        table_dict = json.loads(table_str)
        query_df = pd.DataFrame(**table_dict)

        query_df['env_name'] = [run.config['env_name'] for _ in range(len(query_df))]
        query_df['dataset_name'] = [run.config['dataset_name'] for _ in range(len(query_df))]
        query_df['n_step'] = [run.config['reset_n_step'] for _ in range(len(query_df))]
        query_df['dynamics_type'] = [run.config['dynamics_type'] for _ in range(len(query_df))]
        query_df['uncertainty_test_type'] = [run.config['uncertainty_test_type'] for _ in range(len(query_df))]
        query_df['deterministic'] = [run.config['deterministic'] for _ in range(len(query_df))]
        query_df['constant_prior'] = [run.config['constant_prior'] for _ in range(len(query_df))]
        ensemble_data_df.append(query_df)

        table_file = wandb.restore(run.summary.get('horizon-data').get("path"),
                                   run_path='/'.join(run.path))
        table_str = table_file.read()
        table_dict = json.loads(table_str)
        query_df = pd.DataFrame(**table_dict)

        query_df['env_name'] = [run.config['env_name'] for _ in range(len(query_df))]
        query_df['dataset_name'] = [run.config['dataset_name'] for _ in range(len(query_df))]
        query_df['n_step'] = [run.config['reset_n_step'] for _ in range(len(query_df))]
        query_df['dynamics_type'] = [run.config['dynamics_type'] for _ in range(len(query_df))]
        query_df['uncertainty_test_type'] = [run.config['uncertainty_test_type'] for _ in range(len(query_df))]
        query_df['deterministic'] = [run.config['deterministic'] for _ in range(len(query_df))]
        query_df['constant_prior'] = [run.config['constant_prior'] for _ in range(len(query_df))]
        horizon_data_df.append(query_df)

    ensemble_data_df = pd.concat(ensemble_data_df)
    horizon_data_df = pd.concat(horizon_data_df)
    uncertainty_test_types = np.unique(ensemble_data_df['uncertainty_test_type'].values)
    envs = np.unique(ensemble_data_df['env_name'].values)

    # overall fig
    for uncertainty_test_type in ['ensemble-voting']:
        for env in envs.tolist() + ['Overall']:

            # Create Base of plots
            uncertainty_plots = {
                _name: {'dynamics_type': plt.subplots(1, sub_plot_count, figsize=(10, 2.5), squeeze=False,
                                                      layout='constrained'),
                        'dataset_name': plt.subplots(1, sub_plot_count, figsize=(10, 2.5), squeeze=False,
                                                     layout='constrained'),
                        'horizon': plt.subplots(1, sub_plot_count, figsize=(10, 2.5), squeeze=False,
                                                layout='constrained'),
                        'ensemble_count': plt.subplots(1, sub_plot_count, figsize=(10, 2.5), squeeze=False,
                                                       layout='constrained'),
                        'deterministic': plt.subplots(1, sub_plot_count, figsize=(10, 2.5), squeeze=False,
                                                      layout='constrained'),
                        'constant_prior': plt.subplots(1, sub_plot_count, figsize=(10, 2.5), squeeze=False,
                                                       layout='constrained'),
                        'n_step': plt.subplots(1, sub_plot_count, figsize=(10, 2.5), squeeze=False,
                                               layout='constrained')}
                for _name, sub_plot_count in [('accuracy_vs_abstain', 4)]}

            if env == 'Overall':
                ensemble_type_data = ensemble_data_df[ensemble_data_df['uncertainty_test_type']
                                                      == uncertainty_test_type]
                horizon_type_data = horizon_data_df[horizon_data_df['uncertainty_test_type']
                                                    == uncertainty_test_type]
            else:
                ensemble_type_data = ensemble_data_df[(ensemble_data_df['uncertainty_test_type']
                                                       == uncertainty_test_type) &
                                                      (ensemble_data_df['env_name'] == env)]
                horizon_type_data = horizon_data_df[(horizon_data_df['uncertainty_test_type'] ==
                                                     uncertainty_test_type) &
                                                    (horizon_data_df['env_name'] == env)]

            if uncertainty_test_type == 'ensemble-voting':
                conf_attr_name = 'confidence_threshold'
            else:
                conf_attr_name = 'confidence_level'

            # ##################
            # get labels
            # #################
            conf_thresholds = np.unique(ensemble_type_data[conf_attr_name])
            dynamics_types = sorted(np.unique(ensemble_type_data['dynamics_type']))
            dataset_names = sorted(np.unique(ensemble_type_data['dataset_name']))
            query_horizons = sorted(np.unique(horizon_type_data['horizon']))
            ensemble_counts = sorted(np.unique(ensemble_type_data['ensemble_count']))
            deterministic_dynamics = sorted(np.unique(ensemble_type_data['deterministic']))
            constant_priors = sorted(np.unique(ensemble_type_data['constant_prior']))
            n_steps = sorted(np.unique(ensemble_type_data['n_step']))

            for entity_name, entity_itr, type_data in \
                    [('dataset_name', dataset_names, ensemble_type_data),
                     ('dynamics_type', dynamics_types, ensemble_type_data),
                     ('horizon', query_horizons, horizon_type_data),
                     ('ensemble_count', ensemble_counts, ensemble_type_data),
                     ('deterministic', deterministic_dynamics, ensemble_type_data),
                     ('constant_prior', constant_priors, ensemble_type_data),
                     ('n_step', n_steps, ensemble_type_data)]:

                entity_fig, entity_axs = uncertainty_plots['accuracy_vs_abstain'][entity_name]

                if len(entity_itr) > 10:
                    _color_picker = list(plt.get_cmap('tab20').colors)
                else:
                    _color_picker = list(plt.get_cmap('tab10').colors)

                for entity_i in entity_itr:
                    if len(type_data[type_data[entity_name] == entity_i]) == 0:
                        continue
                    accuracy_mean_data = []
                    accuracy_std_data = []
                    abstain_mean_data = []
                    abstain_std_data = []
                    non_abstain_ratios = []
                    for conf_threshold in conf_thresholds:
                        conf_data = type_data[(type_data[entity_name] == entity_i) &
                                              (type_data[conf_attr_name] == conf_threshold)]
                        accuracy_mean = np.mean(conf_data['accuracy'].values)
                        abstain_mean = np.mean(conf_data['abstain'].values)
                        accuracy_mean_data.append(accuracy_mean)
                        abstain_mean_data.append(abstain_mean)

                        accuracy_std = np.std(conf_data['accuracy'].values)
                        abstain_std = np.std(conf_data['abstain'].values)

                        accuracy_std_data.append(accuracy_std)
                        abstain_std_data.append(abstain_std)

                        # add chart for test-query count
                        non_abstain_count = np.mean(conf_data['non_abstain_count'])
                        abstain_count = np.mean(conf_data['abstain_count'])
                        non_abstain_ratios.append(non_abstain_count / (non_abstain_count + abstain_count))

                    _color = _color_picker.pop(0)
                    entity_axs[0][0].plot(conf_thresholds, accuracy_mean_data, color=_color, label=entity_i)
                    # entity_axs[0].errorbar(conf_thresholds, accuracy_mean_data, accuracy_std_data,
                    #                        linestyle='None', marker='^',color=_color)
                    entity_axs[0][0].set_xlabel(conf_attr_name.replace('_', '-'))
                    entity_axs[0][0].set_ylabel('accuracy')

                    entity_axs[0][1].plot(conf_thresholds, abstain_mean_data, color=_color, label=entity_i)
                    # entity_axs[1].errorbar(conf_thresholds, abstain_mean_data, abstain_std_data,
                    #                        linestyle='None', marker='^', color=_color)
                    entity_axs[0][1].set_xlabel(conf_attr_name.replace('_', '-'))
                    entity_axs[0][1].set_ylabel('abstain')

                    entity_axs[0][2].plot(abstain_mean_data, accuracy_mean_data, color=_color, label=entity_i)
                    entity_axs[0][2].set_xlabel('abstain')
                    entity_axs[0][2].set_ylabel('accuracy')

                    # non_abstain_count:
                    entity_axs[0][3].plot(conf_thresholds, non_abstain_ratios, color=_color, label=entity_i)
                    entity_axs[0][3].set_xlabel(conf_attr_name.replace('_', '-'))
                    entity_axs[0][3].set_ylabel('non-abstain-count (%)')

            uncertainty_plots['accuracy_vs_abstain']['dynamics_type'][1][0, 0]. \
                legend(dynamics_types).set_visible(False)
            uncertainty_plots['accuracy_vs_abstain']['dataset_name'][1][0, 0].legend(
                dataset_names).set_visible(False)
            uncertainty_plots['accuracy_vs_abstain']['horizon'][1][0, 0].legend(
                query_horizons).set_visible(False)
            # uncertainty_plots['accuracy_vs_abstain']['n_step'][1][0, 0].legend(
            #     query_horizons).set_visible(False)
            uncertainty_plots['accuracy_vs_abstain']['ensemble_count'][1][0, 0].legend(
                ensemble_counts).set_visible(False)
            uncertainty_plots['accuracy_vs_abstain']['deterministic'][1][0, 0].legend(
                labels=['deterministic' if x is True else 'stochastic' for x in
                        deterministic_dynamics]).set_visible(False)
            uncertainty_plots['accuracy_vs_abstain']['constant_prior'][1][0, 0].legend(
                labels=['constant_prior' if x is True else 'without constant prior' for x in
                        deterministic_dynamics]).set_visible(False)
            uncertainty_plots['accuracy_vs_abstain']['n_step'][1][0, 0].legend(
                labels=n_steps).set_visible(False)

            for _name in uncertainty_plots:
                _path = os.path.join(os.getcwd(), 'uncertainty_test_plots', uncertainty_test_type, _name, env)
                os.makedirs(_path, exist_ok=True)
                for (_fig, _axs), _name, _fig_title, ncol, bbox_to_anchor in [
                    (uncertainty_plots[_name]['dynamics_type'], 'dynamics',
                     "Comparison of different dynamics models", 1, (0.7, -0.1)),
                    (uncertainty_plots[_name]['dataset_name'], 'dataset',
                     "Comparison between Datasets", 1, (0.9, -0.1)),
                    (uncertainty_plots[_name]['ensemble_count'], 'ensemble_count',
                     "Comparison between Ensemble Sizes", 1, (0.9, -0.1)),
                    (uncertainty_plots[_name]['n_step'], 'n_step',
                     "Comparison between n-step models", 1, (0.9, -0.1)),
                    (uncertainty_plots[_name]['deterministic'], 'deterministic',
                     "Comparison between Deterministic(True)/Stochastic(False)  Model", 1, (0.9, -0.1)),
                    (uncertainty_plots[_name]['constant_prior'], 'constant_prior',
                     "Comparison between Constant Prior(True)/ No Constant Prior (False)  Model", 1, (0.9, -0.1)),
                    (uncertainty_plots[_name]['horizon'], 'horizon',
                     "Comparison between different horizon lengths", 1, (0.9, -0.1))]:

                    _axs[0, 0].legend().set_visible(False)
                    handles, labels = _axs[0, 0].get_legend_handles_labels()

                    kwargs = {}
                    if ncol is not None:
                        kwargs['ncol'] = min(len(labels), 5)
                    if bbox_to_anchor is not None:
                        kwargs['bbox_to_anchor'] = bbox_to_anchor

                    _fig.suptitle(_fig_title)
                    _fig.legend(handles, labels, **kwargs)
                    _fig.savefig(os.path.join(_path, '{}.png'.format(_name)), bbox_inches="tight")
                    plt.close(_fig)

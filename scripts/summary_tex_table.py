import json
import os
import pickle

import numpy as np
import pandas as pd
import wandb


def camel_case_split(str):
    start_idx = [i for i, e in enumerate(str)
                 if e.isupper()] + [len(str)]

    start_idx = [0] + start_idx
    return [str[x: y] for x, y in zip(start_idx, start_idx[1:])]


def score(metric_info):
    return "$" + (str(round(metric_info['mean'], 3))
                  + "\pm"
                  + str(round(metric_info['std'], 2))) + "$"


def metrics(data, key):
    info = {}
    for _num in np.unique(data[key].values):
        inner_data = data[data[key] == _num]
        env_names = np.unique(inner_data['env_name'].values)
        for env_name in env_names:
            loss = inner_data[inner_data['env_name']
                              == env_name]['loss'].values
            aurcc = inner_data[inner_data['env_name']
                               == env_name]['aurcc'].values
            rpp = inner_data[inner_data['env_name'] == env_name]['rpp'].values
            cr_10 = inner_data[inner_data['env_name'] == env_name][
                'cr_10'].values
            if env_name not in info:
                info[env_name] = {}

            info[env_name][_num] = {'aurcc': {'mean': aurcc.mean(),
                                              'std': aurcc.std(),
                                              'n': len(aurcc)},
                                    'rpp': {'mean': rpp.mean(),
                                            'std': rpp.std(),
                                            'n': len(aurcc)},
                                    'cr_10': {'mean': cr_10.mean(),
                                              'std': cr_10.std(),
                                              'n': len(cr_10)},
                                    'loss': {'mean': loss.mean(),
                                             'std': loss.std(),
                                             'n': len(loss)}
                                    }
    return info


def prettify_env_name(env_name):
    if "d4rl:maze2d-" in env_name:
        x = env_name.split("d4rl:maze2d-")[1].split("-v")[0]
        return x
    elif "Walker" in env_name:
        return "Walker\\\\2d"
    else:
        x = env_name.split("-v")[0]
        return "\\\\".join(camel_case_split(x)[1:])


def prettify_category_name(name):
    if name == 'ensemble-voting':
        return 'ev'
    elif name == 'unpaired-confidence-interval':
        return 'u-pci'
    elif name == 'paired-confidence-interval':
        return 'pci'
    else:
        return name


def sr_coverage_plot(risk, coverage, category_name, plot_name, path):
    pass


def latex_table(info_dict, category_name, table_name, path):
    tex = "\\begin{table}[t!]" + "\n" + \
          "\caption{Evaluation metrics for  \emph{" + table_name + "}" + \
          " comparison in \emph{" + category_name + "} environments } " + "\n" + \
          "\label{table:" + category_name + '-' + table_name + "}" + "\n"
    tex += "\\begin{center} " + "\n" + \
           "\\begin{footnotesize} " + "\n" + \
           "\\begin{sc} " + "\n" + \
           "\\begin{tabular}{|l | c | c | c| c | c | c |}" + "\n" + \
           "\\toprule" + "\n"
    tex += "Env. & " + table_name + " & AURCC$(\\downarrow)$ &" \
                                    " RPP$(\\downarrow)$ &" \
                                    " $CR_K(\\uparrow)$ & " \
                                    " loss$(\\downarrow)$ & " \
                                    "runs \\\\" + '\n'
    tex += "\\midrule" + '\n'
    for env_i, env_name in enumerate(info_dict.keys()):
        if len(info_dict[env_name]) > 1:
            tex += "\multirow{" + str(len(info_dict[env_name])) \
                   + "}{3.6em}{" + str(prettify_env_name(env_name)) + "}  "
        else:
            tex += str(prettify_env_name(env_name)) + " "
        for cat in info_dict[env_name]:
            tex += "& " \
                   + str(prettify_category_name(cat)) \
                   + " & " + score(info_dict[env_name][cat]['aurcc']) \
                   + " & " + score(info_dict[env_name][cat]['rpp']) \
                   + " & " + score(info_dict[env_name][cat]['cr_10']) \
                   + " & " + score(info_dict[env_name][cat]['loss']) \
                   + " & " + str(info_dict[env_name][cat]['aurcc']['n']) \
                   + " \\\\ \n"
        if env_i == len(info_dict) - 1:
            tex += "\\bottomrule \n"
        else:
            tex += "\\midrule \n"
    tex += "\\end{tabular}" + "\n"
    tex += "\\end{sc}" + "\n"
    tex += "\\end{footnotesize} " + "\n"
    tex += "\\end{center} " + "\n"
    tex += "\\end{table}" + "\n"

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'metrics.tex'), 'w') as f:
        f.write(tex)
    return tex


def compress_rc_data(df):
    risk = []
    coverage = []
    for tau in df['taus'].unique():
        risk.append(df['risk'][df['taus'] == tau].values.tolist())
        coverage.append(df['coverage'][df['taus'] == tau].values.tolist())

    return np.mean(risk, axis=1), np.std(risk, axis=1), \
           np.mean(coverage, axis=1), np.std(coverage, axis=1)


def generate_graphics(category_name, data_df, sr_coverage_df):
    root_dir = os.path.join(os.getcwd(), category_name)

    # base-model row
    base_prior_scale = 0
    base_deter = True
    base_mixture = False
    base_dyn_type = 'feed-forward'
    base_normalize = True
    base_clip_obs = True
    base_clip_reward = True
    base_uncertainty_type = 'ensemble-voting'

    base_data = data_df[(data_df['constant_prior_scale'] == base_prior_scale)
                        & (data_df['deterministic'] == base_deter)
                        & (data_df['mixture'] == base_mixture)
                        & (data_df['dynamics_type'] == base_dyn_type)
                        & (data_df['normalize'] == base_normalize)
                        & (data_df['clip_obs'] == base_clip_obs)
                        & (data_df['clip_reward'] == base_clip_reward)
                        & (data_df['uncertainty_type'] ==
                           base_uncertainty_type)
                        & (np.isnan(data_df['horizon'].values))]

    base_sr_coverage_data = sr_coverage_df[
        (sr_coverage_df['constant_prior_scale']
         == base_prior_scale)
        & (sr_coverage_df['deterministic'] == base_deter)
        & (sr_coverage_df['mixture'] == base_mixture)
        & (sr_coverage_df['dynamics_type'] == base_dyn_type)
        & (sr_coverage_df['normalize'] == base_normalize)
        & (sr_coverage_df['clip_obs'] == base_clip_obs)
        & (sr_coverage_df['clip_reward'] == base_clip_reward)
        & (sr_coverage_df['uncertainty_type'] ==
           base_uncertainty_type)
        & (np.isnan(sr_coverage_df['horizon'].values))]

    # ensemble-count
    ensemble_count_info = metrics(base_data, 'ensemble_count')
    latex_table(ensemble_count_info, category_name, 'ensemble-count',
                os.path.join(root_dir, 'ensemble-count'))
    risk, risk_stds, coverage, coverage_std = compress_rc_data(sr_coverage_df)
    sr_coverage_plot(ensemble_count_info, category_name, 'ensemble-count',
                     os.path.join(root_dir, 'ensemble-count'))

    dataset_quality_info = metrics(base_data, 'dataset_name')
    latex_table(dataset_quality_info, category_name, 'dataset-quality',
                os.path.join(root_dir, 'dataset-quality'))

    # horizon-type
    hor_data = data_df[(data_df['constant_prior_scale'] == base_prior_scale)
                       & (data_df['deterministic'] == base_deter)
                       & (data_df['mixture'] == base_mixture)
                       & (data_df['dynamics_type'] == base_dyn_type)
                       & (data_df['normalize'] == base_normalize)
                       & (data_df['clip_obs'] == base_clip_obs)
                       & (data_df['clip_reward'] == base_clip_reward)
                       & (data_df['uncertainty_type'] ==
                          base_uncertainty_type)
                       & (~np.isnan(data_df['horizon'].values))]
    horizon_info = metrics(hor_data, 'horizon')
    latex_table(horizon_info, category_name, 'horizon',
                os.path.join(root_dir, 'horizon'))

    # ##############################
    # constant-prior
    # ##############################
    prior_data = data_df[(data_df['deterministic'] == base_deter)
                         & (data_df['mixture'] == base_mixture)
                         & (data_df['dynamics_type'] == base_dyn_type)
                         & (data_df['normalize'] == base_normalize)
                         & (data_df['clip_obs'] == base_clip_obs)
                         & (data_df['clip_reward'] == base_clip_reward)
                         & (data_df['uncertainty_type'] ==
                            base_uncertainty_type)
                         & np.isnan(data_df['horizon'].values)]
    prior_info = metrics(prior_data, 'constant_prior_scale')
    latex_table(prior_info, category_name, 'prior-scale',
                os.path.join(root_dir, 'prior-scale'))

    # ##############################
    # deterministic
    # ##############################

    deter_data = data_df[(data_df['constant_prior_scale'] == base_prior_scale)
                         & (data_df['mixture'] == base_mixture)
                         & (data_df['dynamics_type'] == base_dyn_type)
                         & (data_df['normalize'] == base_normalize)
                         & (data_df['clip_obs'] == base_clip_obs)
                         & (data_df['clip_reward'] == base_clip_reward)
                         & (data_df['uncertainty_type'] ==
                            base_uncertainty_type)
                         & np.isnan(data_df['horizon'].values)]
    deterministic_info = metrics(deter_data, 'deterministic')
    latex_table(deterministic_info, category_name, 'deterministic',
                os.path.join(root_dir, 'deterministic'))

    # ##############################
    # dynamics-type
    # ##############################
    dyn_data = data_df[(data_df['constant_prior_scale'] == base_prior_scale)
                       & (data_df['deterministic'] == base_deter)
                       & (data_df['mixture'] == base_mixture)
                       & (data_df['normalize'] == base_normalize)
                       & (data_df['clip_obs'] == base_clip_obs)
                       & (data_df['clip_reward'] == base_clip_reward)
                       & (data_df['uncertainty_type'] ==
                          base_uncertainty_type)
                       & np.isnan(data_df['horizon'])]
    dynamics_type_info = metrics(dyn_data, 'dynamics_type')
    latex_table(dynamics_type_info, category_name, 'dynamics-type',
                os.path.join(root_dir, 'dynamics-type'))

    # ##############################
    # mixture
    # ##############################
    mix_data = data_df[(data_df['constant_prior_scale'] == base_prior_scale)
                       & (data_df['deterministic'] == base_deter)
                       & (data_df['dynamics_type'] == base_dyn_type)
                       & (data_df['normalize'] == base_normalize)
                       & (data_df['clip_obs'] == base_clip_obs)
                       & (data_df['clip_reward'] == base_clip_reward)
                       & (data_df['uncertainty_type'] ==
                          base_uncertainty_type)
                       & np.isnan(data_df['horizon'])]
    mixture_info = metrics(mix_data, 'mixture')
    latex_table(mixture_info, category_name, 'mixture',
                os.path.join(root_dir, 'mixture'))

    # ##############################
    # uncertainty-test type
    # ##############################
    type_data = data_df[(data_df['constant_prior_scale'] == base_prior_scale)
                        & (data_df['deterministic'] == base_deter)
                        & (data_df['mixture'] == base_mixture)
                        & (data_df['dynamics_type'] == base_dyn_type)
                        & (data_df['normalize'] == base_normalize)
                        & (data_df['clip_obs'] == base_clip_obs)
                        & (data_df['clip_reward'] == base_clip_reward)
                        & (np.isnan(data_df['horizon'].values))]
    mixture_info = metrics(type_data, 'uncertainty_type')
    latex_table(mixture_info, category_name, 'uncertainty-type',
                os.path.join(root_dir, 'uncertainty-type'))

    # ##############################
    # normalization
    # ##############################
    norm_data = data_df[(data_df['constant_prior_scale'] == base_prior_scale)
                        & (data_df['deterministic'] == base_deter)
                        & (data_df['mixture'] == base_mixture)
                        & (data_df['dynamics_type'] == base_dyn_type)
                        & (data_df['clip_obs'] == base_clip_obs)
                        & (data_df['clip_reward'] == base_clip_reward)
                        & (data_df['uncertainty_type'] ==
                           base_uncertainty_type)
                        & (np.isnan(data_df['horizon'].values))]
    norm_info = metrics(norm_data, 'normalize')
    latex_table(norm_info, category_name, 'normalize',
                os.path.join(root_dir, 'normalization'))


def main():
    # combine data from multiple runs into a single dataframe
    api = wandb.Api()
    eval_metrics_df = []
    sr_coverage_df = []
    run_i = 0
    for run in api.runs('koulanurag/opcc-baselines-uncertainty-test',
                        filters={'state': 'finished'}):
        print(run_i, run)
        run_i += 1

        # restore evaluation metrics
        table_file = wandb.restore(run.summary.get('eval-metrics').get("path"),
                                   run_path='/'.join(run.path))
        table_str = table_file.read()
        table_dict = json.loads(table_str)
        query_df = pd.DataFrame(**table_dict)

        # restore rcc-curve data
        _coverage_dict = pickle.load(open(wandb.restore("sr_coverage_dict.pkl",
                                                        "/".join(run.path)).name, 'rb'))
        _coverage_df = []
        for count in _coverage_dict:
            for horizon, count_horizon_info in _coverage_dict[count].items():
                _coverage_df.append(pd.DataFrame.from_dict(
                    {**count_horizon_info,
                     **{'ensemble-count': [count for _ in
                                           range(len(count_horizon_info['taus']))],
                        'horizon': [horizon for _ in
                                    range(len(count_horizon_info['taus']))]}}))
        _coverage_df = pd.concat(_coverage_df)

        for _df in [query_df, _coverage_df]:
            _df['env_name'] = [run.config['env_name']
                               for _ in range(len(_df))]
            _df['dataset_name'] = [run.config['dataset_name']
                                   for _ in range(len(_df))]

            # dynamics factors
            _df['dynamics_type'] = [run.config['dynamics_type']
                                    for _ in range(len(_df))]
            _df['deterministic'] = [run.config['deterministic']
                                    for _ in range(len(_df))]
            _df['constant_prior_scale'] = [run.config['constant_prior_scale']
                                           for _ in range(len(_df))]
            _df['dynamics_seed'] = [run.config['dynamics_seed']
                                    for _ in range(len(_df))]
            _df['normalize'] = [run.config['normalize']
                                for _ in range(len(_df))]

            # evaluation factors
            _df['mixture'] = [run.config['mixture']
                              for _ in range(len(_df))]
            _df['clip_obs'] = [run.config['clip_obs']
                               for _ in range(len(_df))]
            _df['clip_reward'] = [run.config['clip_reward']
                                  for _ in range(len(_df))]

            # uncertainty factors
            _df['uncertainty_type'] = [run.config['uncertainty_test_type']
                                       for _ in range(len(_df))]

        eval_metrics_df.append(query_df)
        sr_coverage_df.append(_coverage_df)

        if run_i > 20:
            break

    eval_metrics_df = pd.concat(eval_metrics_df)
    sr_coverage_df = pd.concat(sr_coverage_df)
    eval_metrics_df.to_pickle('eval_metrics_df.pkl')
    sr_coverage_df.to_pickle('sr_coverage_df.pkl')

    eval_metrics_df = pd.read_pickle('eval_metrics_df.pkl')
    sr_coverage_df = pd.read_pickle('sr_coverage_df.pkl')

    gym_mujoco_envs = ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2']
    maze2d_envs = ['d4rl:maze2d-open-v0', 'd4rl:maze2d-umaze-v1',
                   'd4rl:maze2d-medium-v1', 'd4rl:maze2d-large-v1']
    for category_name, env_names in [('maze', maze2d_envs),
                                     ('gym-mujoco', gym_mujoco_envs)]:
        if len(eval_metrics_df[eval_metrics_df['env_name'].
                isin(env_names)]) > 0:
            generate_graphics(category_name,
                              eval_metrics_df[eval_metrics_df['env_name'].
                              isin(env_names)],
                              sr_coverage_df[sr_coverage_df['env_name'].
                              isin(env_names)])


if __name__ == '__main__':
    main()

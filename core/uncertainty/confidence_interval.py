import scipy
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def paired_confidence_interval(pred_a, pred_b, target, target_return_a,
                               target_return_b, conf_level_interval,
                               dict_to_add=None):
    """
    :param pred_a: numpy array of shape (batch, ensemble_count) having
                   q-value estimates of query-a
    :param pred_b: numpy array of shape (batch, ensemble_count) having
                   q-value estimates of query-b
    :param target: boolean numpy array of shape (batch,) having target label
    """

    uncertainty_df = []
    delta_pred = pred_a - pred_b
    not_abstain_conf_level = np.array([None for _ in range(len(delta_pred))])
    base_res_low = np.array([None for _ in range(len(pred_a))])
    base_res_high = np.array([None for _ in range(len(pred_a))])

    for conf_level in np.arange(0, 1.01, conf_level_interval):
        res_low = np.zeros(len(pred_a))
        res_high = np.zeros(len(pred_a))

        for idx in range(len(delta_pred)):
            conf_interval = scipy.stats.t.interval(conf_level,
                                                   len(delta_pred[idx]) - 1,
                                                   np.mean(delta_pred[idx]),
                                                   scipy.stats.sem(delta_pred[idx]) + 1e-3)
            res_low[idx], res_high[idx] = conf_interval

        pred_label = np.zeros(len(delta_pred))
        pred_label[res_high < 0] = 1  # true
        pred_label[res_low > 0] = 0  # false
        # abstain
        pred_label[np.logical_and(res_low < 0, (res_high > 0))] = -1
        not_abstain_conf_level[pred_label != -1] = conf_level
        if len(pred_label[pred_label != -1]) > 0:
            accuracy = (pred_label[pred_label != -1]
                        == target[pred_label != -1]).mean()
        else:
            accuracy = 1.0
        abstain = len(pred_label[pred_label == -1]) / len(pred_label)

        # filter flags
        accept_idx = pred_label != -1
        abstain_idx = pred_label == -1

        base_res_low[accept_idx] = res_low[accept_idx]
        base_res_high[accept_idx] = res_high[accept_idx]

        tn, fp, fn, tp = confusion_matrix(target[accept_idx],
                                          pred_label[accept_idx],
                                          labels=[False, True]).ravel()

        _log = {'abstain': abstain,
                'accuracy': accuracy,
                'inaccuracy': 1 - accuracy,
                'accuracy_count': tp + tn,
                'inaccuracy_count': fp + fn,
                'abstain_count': len(pred_label[abstain_idx]),
                'non_abstain_count': len(pred_label[accept_idx]),
                'query_count': len(pred_label),
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'true_negative': tn,
                'confidence_level': conf_level,
                'value_regret_risk': np.abs(target_return_a.values[accept_idx] -
                                            target_return_b.values[accept_idx]).mean()}
        if dict_to_add is not None:
            _log = {**_log, **dict_to_add}

        uncertainty_df.append(
            pd.DataFrame({_k: [_v] for _k, _v in _log.items()}))

    pred_label = np.array([None for _ in range(len(pred_a))])
    pred_label[base_res_high < 0] = 1  # true
    pred_label[base_res_low > 0] = 0  # false

    rpp = np.logical_and(
        np.expand_dims(not_abstain_conf_level, 1).transpose() < np.expand_dims(not_abstain_conf_level, 1),
        np.expand_dims(pred_label, 1).transpose() < np.expand_dims(pred_label, 1))
    rpp_df = {**{k: [v] for k, v in dict_to_add.items()},
              **{'rpp': [rpp.mean()]}}
    rpp_df = [pd.DataFrame(data=rpp_df)]

    return uncertainty_df, rpp_df


def unpaired_confidence_interval(pred_a, pred_b, target, target_return_a,
                                 target_return_b, conf_level_interval,
                                 dict_to_add=None):
    """
    :param pred_a: numpy array of shape (batch, ensemble_count) having
                   q-value estimates of query-a
    :param pred_b: numpy array of shape (batch, ensemble_count) having
                   q-value estimates of query-b
    :param target: boolean numpy array of shape (batch,) having target label
    """

    uncertainty_df = []
    not_abstain_conf_level = np.array([None for _ in range(len(pred_a))])

    base_res_low_a = np.array([None for _ in range(len(pred_a))])
    base_res_high_a = np.array([None for _ in range(len(pred_a))])
    base_res_low_b = np.array([None for _ in range(len(pred_a))])
    base_res_high_b = np.array([None for _ in range(len(pred_a))])

    for conf_level in np.arange(0, 1.01, conf_level_interval):
        res_low_a = np.zeros(len(pred_a))
        res_high_a = np.zeros(len(pred_a))
        res_low_b = np.zeros(len(pred_a))
        res_high_b = np.zeros(len(pred_a))

        for idx in range(len(pred_a)):
            a_conf_interval = scipy.stats.t.interval(
                conf_level,
                len(pred_a[idx]) - 1,
                np.mean(pred_a[idx]),
                scipy.stats.sem(pred_a[idx] + 1e-3))

            res_low_a[idx], res_high_a[idx] = a_conf_interval

            b_conf_interval = scipy.stats.t.interval(
                conf_level,
                len(pred_b[idx]) - 1,
                np.mean(pred_b[idx]),
                scipy.stats.sem(pred_b[idx] + 1e-3))

            res_low_b[idx], res_high_b[idx] = b_conf_interval

        pred_label = -1 * np.ones(len(pred_a))  # abstain
        pred_label[res_high_a < res_low_b] = 1  # true
        pred_label[res_low_a > res_high_b] = 0  # false

        if len(pred_label[pred_label != -1]) > 0:
            accuracy = (pred_label[pred_label != -1]
                        == target[pred_label != -1]).mean()
            value_regret_risk = np.abs(target_return_a.values[accept_idx]
                                       - target_return_b.values[accept_idx]).mean()
        else:
            accuracy = 1.0
            value_regret_risk = 0
        abstain = len(pred_label[pred_label == -1]) / len(pred_label)

        # filter flags
        accept_idx = pred_label != -1
        abstain_idx = pred_label == -1

        not_abstain_conf_level[accept_idx] = conf_level
        base_res_low_a[accept_idx] = res_low_a[accept_idx]
        base_res_high_a[accept_idx] = res_high_a[accept_idx]
        base_res_low_b[accept_idx] = res_low_b[accept_idx]
        base_res_high_b[accept_idx] = res_high_b[accept_idx]

        tn, fp, fn, tp = confusion_matrix(target[accept_idx],
                                          pred_label[accept_idx],
                                          labels=[False, True]).ravel()

        _log = {'abstain': abstain,
                'accuracy': accuracy,
                'inaccuracy': 1 - accuracy,
                'accuracy_count': tp + tn,
                'inaccuracy_count': fp + fn,
                'abstain_count': len(pred_label[abstain_idx]),
                'non_abstain_count': len(pred_label[accept_idx]),
                'query_count': len(pred_label),
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'true_negative': tn,
                'confidence_level': conf_level,
                'value_regret_risk': value_regret_risk}
        if dict_to_add is not None:
            _log = {**_log, **dict_to_add}

        uncertainty_df.append(
            pd.DataFrame({_k: [_v] for _k, _v in _log.items()}))

    pred_label = np.array([None for _ in range(len(pred_a))])
    pred_label[base_res_high_a < base_res_low_b] = 1  # true
    pred_label[base_res_low_a > base_res_high_b] = 0  # false

    rpp = np.logical_and(
        np.expand_dims(not_abstain_conf_level, 1).transpose() < np.expand_dims(not_abstain_conf_level, 1),
        np.expand_dims(pred_label, 1).transpose() < np.expand_dims(pred_label, 1))
    rpp_df = {**{k: [v] for k, v in dict_to_add.items()},
              **{'rpp': [rpp.mean()]}}
    rpp_df = [pd.DataFrame(data=rpp_df)]
    return uncertainty_df, rpp_df


def confidence_interval(eval_df, ensemble_size_interval: int, num_ensemble: int,
                        step: float = 0.1, paired: bool = True):
    pred_a = [np.expand_dims(eval_df['pred_a_{}'.format(e_i)].values, 1)
              for e_i in range(num_ensemble)]
    pred_a = np.concatenate(pred_a, axis=1)
    pred_b = [np.expand_dims(eval_df['pred_b_{}'.format(e_i)].values, 1)
              for e_i in range(num_ensemble)]
    pred_b = np.concatenate(pred_b, axis=1)
    target = eval_df['target'].values

    # process for ensemble counts
    ensemble_uncertainty_df = []
    ensemble_rpps_df = []
    for ensemble_count in np.arange(min(ensemble_size_interval, num_ensemble),
                                    num_ensemble + 1,
                                    ensemble_size_interval):
        if paired:
            _df, _rpp_df = paired_confidence_interval(pred_a[:, :ensemble_count],
                                                      pred_b[:, :ensemble_count],
                                                      target,
                                                      eval_df['return_a'],
                                                      eval_df['return_b'],
                                                      step,
                                                      {'ensemble_count': ensemble_count})
        else:
            _df, _rpp_df = unpaired_confidence_interval(pred_a[:, :ensemble_count],
                                                        pred_b[:, :ensemble_count],
                                                        target,
                                                        eval_df['return_a'],
                                                        eval_df['return_b'],
                                                        step,
                                                        {'ensemble_count': ensemble_count})
        ensemble_uncertainty_df += _df
        ensemble_rpps_df += _rpp_df

    ensemble_uncertainty_df = pd.concat(ensemble_uncertainty_df,
                                        ignore_index=True)
    ensemble_rpps_df = pd.concat(ensemble_rpps_df,
                                 ignore_index=True)

    # process for horizons
    horizons = eval_df['horizon'].values
    horizon_candidates = np.unique(horizons, axis=0)
    horizon_uncertainty_df = []
    horizon_rpps_df = []

    for horizon in horizon_candidates:
        _filter = horizons == horizon
        if paired:
            _df, _rpp_df = paired_confidence_interval(pred_a[_filter],
                                                      pred_b[_filter],
                                                      target[_filter],
                                                      eval_df['return_a'][_filter],
                                                      eval_df['return_b'][_filter],
                                                      step,
                                                      {'horizon': horizon})
        else:
            _df, _rpp_df = unpaired_confidence_interval(pred_a[_filter],
                                                        pred_b[_filter],
                                                        target[_filter],
                                                        eval_df['return_a'][_filter],
                                                        eval_df['return_b'][_filter],
                                                        step,
                                                        {'horizon': horizon})
        horizon_uncertainty_df += _df
        horizon_rpps_df += _rpp_df
    horizon_uncertainty_df = pd.concat(horizon_uncertainty_df, ignore_index=True)
    horizon_rpps_df = pd.concat(horizon_rpps_df, ignore_index=True)

    return ensemble_uncertainty_df, horizon_uncertainty_df, \
           ensemble_rpps_df, horizon_rpps_df

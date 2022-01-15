import scipy
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def paired_confidence_interval(pred_a, pred_b, target, conf_level_interval,
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

        if len(pred_label[pred_label != -1]) > 0:
            accuracy = (pred_label[pred_label != -1]
                        == target[pred_label != -1]).mean()
        else:
            accuracy = 1.0
        abstain = len(pred_label[pred_label == -1]) / len(pred_label)

        # filter flags
        accept_idx = pred_label != -1
        abstain_idx = pred_label == -1

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
                'confidence_level': conf_level}
        if dict_to_add is not None:
            _log = {**_log, **dict_to_add}

        uncertainty_df.append(
            pd.DataFrame({_k: [_v] for _k, _v in _log.items()}))

    return uncertainty_df


def unpaired_confidence_interval(pred_a, pred_b, target, conf_level_interval,
                                 dict_to_add=None):
    """
    :param pred_a: numpy array of shape (batch, ensemble_count) having
                   q-value estimates of query-a
    :param pred_b: numpy array of shape (batch, ensemble_count) having
                   q-value estimates of query-b
    :param target: boolean numpy array of shape (batch,) having target label
    """

    uncertainty_df = []
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
        else:
            accuracy = 1.0
        abstain = len(pred_label[pred_label == -1]) / len(pred_label)

        # filter flags
        accept_idx = pred_label != -1
        abstain_idx = pred_label == -1

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
                'confidence_level': conf_level}
        if dict_to_add is not None:
            _log = {**_log, **dict_to_add}

        uncertainty_df.append(
            pd.DataFrame({_k: [_v] for _k, _v in _log.items()}))

    return uncertainty_df


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
    for ensemble_count in np.arange(min(ensemble_size_interval, num_ensemble),
                                    num_ensemble + 1,
                                    ensemble_size_interval):
        if paired:
            ensemble_uncertainty_df += paired_confidence_interval(pred_a[:, :ensemble_count],
                                                                  pred_b[:, :ensemble_count],
                                                                  target, step,
                                                                  {'ensemble_count': ensemble_count})
        else:
            ensemble_uncertainty_df += unpaired_confidence_interval(pred_a[:, :ensemble_count],
                                                                    pred_b[:, :ensemble_count],
                                                                    target, step,
                                                                    {'ensemble_count': ensemble_count})
    ensemble_uncertainty_df = pd.concat(ensemble_uncertainty_df,
                                        ignore_index=True)

    # process for horizons
    horizons = eval_df['horizon'].values
    horizon_candidates = np.unique(horizons, axis=0)
    horizon_uncertainty_df = []

    for horizon in horizon_candidates:
        _filter = horizons == horizon
        if paired:
            horizon_uncertainty_df += paired_confidence_interval(pred_a[_filter],
                                                                 pred_b[_filter],
                                                                 target[_filter],
                                                                 step,
                                                                 {'horizon': horizon})
        else:
            horizon_uncertainty_df += unpaired_confidence_interval(pred_a[_filter],
                                                                   pred_b[_filter],
                                                                   target[_filter],
                                                                   step,
                                                                   {'horizon': horizon})
    horizon_uncertainty_df = pd.concat(horizon_uncertainty_df, ignore_index=True)

    return ensemble_uncertainty_df, horizon_uncertainty_df

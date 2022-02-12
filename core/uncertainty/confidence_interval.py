import numpy as np
import pandas as pd
import scipy
from scipy import stats


def paired_confidence_interval(pred_a, pred_b):
    delta_pred = pred_a - pred_b
    pred_label = np.array([0 for _ in range(len(pred_a))])  # default false
    pred_conf = np.zeros(len(pred_a), dtype=float)  # default 0 confidence
    for conf_level in np.arange(0, 1.01, 0.01):
        df = np.ones(len(delta_pred)) * (len(delta_pred[0]) - 1)
        loc = np.mean(delta_pred, axis=1)
        scale = scipy.stats.sem(delta_pred, axis=1) + 1e-3
        conf_interval = scipy.stats.t.interval(conf_level, df, loc, scale)
        res_low, res_high = conf_interval
        accept_filter = ~np.logical_and((res_low < 0), (res_high > 0))
        pred_label[np.logical_and(res_high < 0, accept_filter)] = 1  # true
        pred_label[np.logical_and((res_low > 0), accept_filter)] = 0  # false
        pred_conf[accept_filter] = conf_level

    pred_label = pred_label.astype(bool)
    return pred_label, pred_conf


def unpaired_confidence_interval(pred_a, pred_b):
    pred_label = np.array([0 for _ in range(len(pred_a))])  # default false
    pred_conf = np.zeros(len(pred_a), dtype=float)  # default 0 confidence
    for conf_level in np.arange(0, 1.01, 0.01):
        df = np.ones(len(pred_a)) * (len(pred_a[0]) - 1)
        loc = np.mean(pred_a, axis=1)
        scale = scipy.stats.sem(pred_a, axis=1) + 1e-3
        conf_interval = scipy.stats.t.interval(conf_level, df, loc, scale)
        res_low_a, res_high_a = conf_interval

        df = np.ones(len(pred_b)) * (len(pred_b[0]) - 1)
        loc = np.mean(pred_b, axis=1)
        scale = scipy.stats.sem(pred_b, axis=1) + 1e-3
        conf_interval = scipy.stats.t.interval(conf_level, df, loc, scale)
        res_low_b, res_high_b = conf_interval

        abstain_filter = np.logical_or(np.logical_and(res_low_a <= res_low_b,
                                                      res_low_b <= res_high_a),
                                       np.logical_and(res_low_a <= res_high_b,
                                                      res_high_b <= res_high_a)
                                       )
        accept_filter = ~abstain_filter

        # true
        pred_label[np.logical_and(res_high_a < res_low_b, accept_filter)] = 1
        # false
        pred_label[np.logical_and((res_low_a > res_high_b), accept_filter)] = 0
        pred_conf[accept_filter] = conf_level

    pred_label = pred_label.astype(bool)
    return pred_label, pred_conf


def confidence_interval(eval_df: pd.DataFrame,
                        ensemble_size_interval: int,
                        num_ensemble: int,
                        paired: bool = True):
    # extract query predictions and corresponding targets
    pred_a = [np.expand_dims(eval_df['pred_a_{}'.format(e_i)].values, 1)
              for e_i in range(num_ensemble)]
    pred_a = np.concatenate(pred_a, axis=1)
    pred_b = [np.expand_dims(eval_df['pred_b_{}'.format(e_i)].values, 1)
              for e_i in range(num_ensemble)]
    pred_b = np.concatenate(pred_b, axis=1)
    target = eval_df['target'].values
    target_return_a = eval_df['return_a'].values
    target_return_b = eval_df['return_b'].values
    horizons = eval_df['horizon'].values
    horizon_candidates = np.unique(horizons, axis=0)

    # process data for various ensemble counts and query horizons
    uncertainty_dict = {}
    for ensemble_count in np.arange(min(ensemble_size_interval, num_ensemble),
                                    num_ensemble + 1,
                                    ensemble_size_interval):
        uncertainty_dict[ensemble_count] = {}
        for horizon in horizon_candidates:
            _filter = horizons == horizon

            if paired:
                prediction, confidence = paired_confidence_interval(
                    pred_a[:, :ensemble_count][_filter],
                    pred_b[:, :ensemble_count][_filter])
            else:
                prediction, confidence = unpaired_confidence_interval(
                    pred_a[:, :ensemble_count][_filter],
                    pred_b[:, :ensemble_count][_filter])

            info = {'prediction': prediction,
                    'target': target[_filter],
                    'confidence': confidence,
                    'value_regret': np.abs(target_return_a[_filter]
                                           - target_return_b[_filter])}
            uncertainty_dict[ensemble_count][horizon] = info

    return uncertainty_dict

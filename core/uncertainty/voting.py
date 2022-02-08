import numpy as np
import pandas as pd


def _voting(pred_a, pred_b):
    # confidence estimation
    true_conf = (pred_a < pred_b).astype(int).mean(axis=1)
    false_conf = 1 - true_conf

    pred_label = np.zeros(len(true_conf))
    pred_label[true_conf >= 0.5] = 1
    pred_label = pred_label.astype(bool)
    pred_conf = np.zeros(len(true_conf))

    # we treat confidence at 0.5 as 0
    pred_conf[true_conf >= 0.5] = (true_conf[true_conf >= 0.5] - 0.5) / 0.5
    pred_conf[false_conf > 0.5] = (false_conf[false_conf > 0.5] - 0.5) / 0.5
    return pred_label, pred_conf


def ensemble_voting(eval_df: pd.DataFrame,
                    ensemble_size_interval: int,
                    num_ensemble: int):
    # extract query predictions and corresponding targets
    pred_a = [np.expand_dims(eval_df['pred_a_{}'.format(e_i)].values, 1)
              for e_i in range(num_ensemble)]
    pred_a = np.concatenate(pred_a, axis=1)
    pred_b = [np.expand_dims(eval_df['pred_b_{}'.format(e_i)].values, 1)
              for e_i in range(num_ensemble)]
    pred_b = np.concatenate(pred_b, axis=1)
    target = eval_df['target'].values
    target_return_a = eval_df['return_a'].values
    target_return_b = eval_df['return_a'].values
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
            prediction, confidence = \
                _voting(pred_a[:, :ensemble_count][_filter],
                        pred_b[:, :ensemble_count][_filter])
            info = {'prediction': prediction,
                    'target': target[_filter],
                    'confidence': confidence,
                    'value_regret': np.abs(target_return_a[_filter]
                                           - target_return_b[_filter])}
            uncertainty_dict[ensemble_count][horizon] = info

    return uncertainty_dict

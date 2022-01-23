import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def _voting(pred_a, pred_b, target, conf_interval, dict_to_add=None):
    uncertainty_df = []
    true_conf = (pred_a < pred_b).astype(int).mean(axis=1)
    false_conf = 1 - true_conf

    pred_label = np.zeros(len(true_conf))
    pred_label[true_conf >= 0.5] = 1
    pred_label = pred_label.astype(bool)
    pred_conf = np.zeros(len(true_conf))

    pred_conf[true_conf >= 0.5] = (true_conf[true_conf >= 0.5] - 0.5) / 0.5
    pred_conf[false_conf > 0.5] = (false_conf[false_conf > 0.5] - 0.5) / 0.5

    for confidence in np.arange(0, 1.01, conf_interval):
        accept_idx = pred_conf >= confidence
        abstain_idx = ~ accept_idx

        tn, fp, fn, tp = confusion_matrix(target[accept_idx],
                                          pred_label[accept_idx],
                                          labels=[False, True]).ravel()
        if len(pred_label[accept_idx]) > 0:
            accuracy = (tp + tn) / (tn + fn + tp + fp)
        else:
            accuracy = 1.0
        abstain = len(pred_label[abstain_idx]) / len(pred_label)

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
                'confidence_threshold': confidence}
        if dict_to_add is not None:
            _log = {**_log, **dict_to_add}

        uncertainty_df.append(
            pd.DataFrame({_k: [_v] for _k, _v in _log.items()}))

    rpp = np.logical_and(np.expand_dims(pred_conf, 1).transpose() < np.expand_dims(pred_conf, 1),
                         np.expand_dims(pred_label, 1).transpose() < np.expand_dims(pred_label, 1))
    rpp_df = {**{k: [v] for k, v in dict_to_add.items()},
              **{'rpp': [rpp.mean()]}}
    rpp_df = [pd.DataFrame(data=rpp_df)]
    return uncertainty_df, rpp_df


def ensemble_voting(eval_df, ensemble_size_interval: int, num_ensemble: int,
                    confidence_interval: float = 0.1):
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
        _df, _rpp_df = _voting(pred_a[:, :ensemble_count],
                               pred_b[:, :ensemble_count],
                               target,
                               confidence_interval,
                               {'ensemble_count': ensemble_count})

        ensemble_rpps_df += _rpp_df
        ensemble_uncertainty_df += _df
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
        _df, _rpp_df = _voting(pred_a[_filter],
                               pred_b[_filter],
                               target[_filter],
                               confidence_interval,
                               {'horizon': horizon})
        horizon_uncertainty_df += _df
        horizon_rpps_df += _rpp_df

    horizon_uncertainty_df = pd.concat(horizon_uncertainty_df, ignore_index=True)
    horizon_rpps_df = pd.concat(horizon_rpps_df, ignore_index=True)

    return ensemble_uncertainty_df, horizon_uncertainty_df, \
           ensemble_rpps_df, horizon_rpps_df

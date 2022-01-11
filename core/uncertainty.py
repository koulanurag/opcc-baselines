
def ensemble_voting():
    uncertainty_df = []
    per_query_df = []
    for ensemble_size in np.arange(min(10, args.num_ensemble), args.num_ensemble + 1, 10):
        for horizon_candidate in query_horizon_candidates:
            _horizon_a, _horizon_b = horizon_candidate
            _filter = np.all(query_horizon == horizon_candidate, axis=1)
            true_confidence = (return_a_ensemble[:, :ensemble_size][_filter] <
                               return_b_ensemble[:, :ensemble_size][_filter]).astype(int).mean(axis=1)
            false_confidence = 1 - true_confidence

            prediction = np.zeros(len(true_confidence))
            prediction[true_confidence >= 0.5] = 1
            prediction = prediction.astype(bool)
            prediction_confidence = np.zeros(len(true_confidence))
            prediction_confidence[true_confidence >= 0.5] = (true_confidence[
                                                                 true_confidence >= 0.5] - 0.5) / 0.5
            prediction_confidence[false_confidence > 0.5] = (false_confidence[
                                                                 false_confidence > 0.5] - 0.5) / 0.5
            for confidence in np.arange(0, 1.01, 0.01):
                accept_idx = prediction_confidence >= confidence
                abstain_idx = ~ accept_idx

                tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target[_filter][accept_idx],
                                                                  prediction[accept_idx],
                                                                  labels=[False, True]).ravel()
                if len(prediction[accept_idx]) > 0:
                    accuracy = (tp + tn) / (tn + fn + tp + fp)
                else:
                    accuracy = 1.0
                abstain = len(prediction[abstain_idx]) / len(prediction)
                _log = {'ensemble_size': ensemble_size,
                        'dataset_name': run.config['dataset_name'],
                        'env_name': run.config['env_name'],
                        'horizon_a': _horizon_a,
                        'horizon_b': _horizon_b,
                        'abstain': abstain,
                        'accuracy': accuracy,
                        'inaccuracy': 1 - accuracy,
                        'accuracy_count': tp + tn,
                        'inaccuracy_count': fp + fn,
                        'abstain_count': len(prediction[abstain_idx]),
                        'non_abstain_count': len(prediction[accept_idx]),
                        'query_count': len(prediction),
                        'true_positive': tp,
                        'false_positive': fp,
                        'false_negative': fn,
                        'true_negative': tn,
                        'confidence_threshold': confidence}
                uncertainty_df.append(pd.DataFrame({_k: [_v] for _k, _v in _log.items()}))
                _plot_target_color = np.array(['abstain' for _ in range(len(target[_filter]))])
                _plot_target_color[accept_idx & (target[_filter] != prediction)] = 'False'
                _plot_target_color[accept_idx & (target[_filter] == prediction)] = 'True'

                per_query_df.append(
                    pd.DataFrame({**{'target_a': target_a[_filter].tolist(),
                                     'target_b': target_b[_filter].tolist(),
                                     'confidence': [confidence for _ in range(len(target[_filter]))],
                                     'ensemble_size': [ensemble_size for _ in range(len(target[_filter]))],
                                     'horizon': [tuple(horizon_candidate) for _ in range(len(target[_filter]))],
                                     'result': _plot_target_color.tolist()},
                                  **{'predict_a_{}'.format(e_): return_a_ensemble[:, e_][_filter].tolist()
                                     for e_ in range(ensemble_size)},
                                  **{'predict_b_{}'.format(e_): return_b_ensemble[:, e_][_filter].tolist()
                                     for e_ in range(ensemble_size)}}))
                if args.use_wandb:
                    wandb.log(_log)

            # inaccuracy_auc = area = trapz(
            #     x=uncertainty_df['confidence_threshold'][
            #         uncertainty_df['ensemble_size'] == ensemble_size].values,
            #     y=uncertainty_df['inaccuracy'][uncertainty_df['ensemble_size'] == ensemble_size].values)
            if args.use_wandb:
                wandb.log({  # 'inaccuracy_auc': inaccuracy_auc,
                    'ensemble_size': ensemble_size,
                    'horizon_a': _horizon_a,
                    'horizon_b': _horizon_b,
                    'query_horizon': (_horizon_a, _horizon_b)})

    uncertainty_df = pd.concat(uncertainty_df, ignore_index=True)
    per_query_df = pd.concat(per_query_df, ignore_index=True)
    if args.use_wandb:
        # fig = px.scatter_3d(per_query_df[per_query_df['ensemble_size'] == args.num_ensemble],
        #                     x='target_a', y='target_b', z='confidence', color='result', symbol='horizon',
        #                     color_discrete_map={'True': 'green', 'False': 'red', 'abstain': 'blue'})
        per_query_vis_log = {}
        for confidence in np.arange(0, 1.01, 0.25):
            fig_query_vis = px.scatter(per_query_df[(per_query_df['ensemble_size'] == args.num_ensemble) &
                                                    (per_query_df['confidence'] == confidence)],
                                       x='target_a', y='target_b', color='result', symbol='horizon',
                                       color_discrete_map={'True': 'green', 'False': 'red', 'abstain': 'blue'},
                                       title="confidence={}".format(round(confidence, 2)))
            per_query_vis_log['per-query-confidence-vis/confidence={}'.format(confidence)] = fig_query_vis
        wandb.log(per_query_vis_log)

        # _tmp_dir = os.path.join(args.result_dir, 'tmp', str(random.random()))
        # os.makedirs(_tmp_dir, exist_ok=True)
        # uncertainty_df_path = os.path.join(_tmp_dir, 'uncertainty_df.p')
        # per_query_df_path = os.path.join(_tmp_dir, 'per_query_df.p')
        # pickle.dump(uncertainty_df, open(uncertainty_df_path, 'wb'))
        # pickle.dump(per_query_df, open(per_query_df_path, 'wb'))
        #
        # wandb.save( glob_str=os.path.abspath(uncertainty_df_path), policy='now')
        # wandb.save(glob_str=os.path.abspath(per_query_df_path), policy='now')

        uncertainty_df_table = wandb.Table(dataframe=uncertainty_df)
        # per_query_df_table = wandb.Table(dataframe=per_query_df)
        wandb.log({'uncertainty-data': uncertainty_df_table})
        # 'per-query-data': per_query_df_table})
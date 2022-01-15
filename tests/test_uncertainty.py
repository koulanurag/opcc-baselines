import pandas as pd
import pytest


def test_ensemble_voting():
    from core.uncertainty import ensemble_voting
    import numpy as np
    import policybazaar
    import cque

    ensemble_size_interval = 5
    num_ensemble = 4
    confidence_interval = 0.1
    env_name = 'Hopper-v2'

    queries = cque.get_queries(env_name)
    (policy_a_id, policy_b_id) = list(queries.keys())[0]
    query_batch = queries[(policy_a_id, policy_b_id)]
    policy_a, _ = policybazaar.get_policy(*policy_a_id)

    return_a = query_batch['info']['return_a']
    return_b = query_batch['info']['return_b']
    target = query_batch['target']
    horizon = query_batch['horizon']

    # #########################################################
    # accuracy should be 1 given ground truth estimates
    # #########################################################
    pred_a = np.repeat(np.expand_dims(np.array(return_a), 1),
                       num_ensemble, axis=1)
    pred_b = np.repeat(np.expand_dims(np.array(return_b), 1),
                       num_ensemble, axis=1)

    eval_df = pd.DataFrame(np.concatenate((pred_a, pred_b,
                                           np.expand_dims(horizon, 1),
                                           np.expand_dims(target, 1)), 1),
                           columns=["pred_a_{}".format(i) for i in range(num_ensemble)]
                                   + ["pred_b_{}".format(i) for i in range(num_ensemble)]
                                   + ['horizon', 'target'])
    ensemble_df, horizon_df = ensemble_voting(eval_df,
                                              ensemble_size_interval,
                                              num_ensemble,
                                              confidence_interval)
    assert (ensemble_df['accuracy'].values == 1).all()
    assert (horizon_df['accuracy'].values == 1).all()

    # #########################################################
    # accuracy should be 0 given swap of ground truth estimates
    # #########################################################
    pred_b = np.repeat(np.expand_dims(np.array(return_a), 1),
                       num_ensemble, axis=1)
    pred_a = np.repeat(np.expand_dims(np.array(return_b), 1),
                       num_ensemble, axis=1)

    eval_df = pd.DataFrame(np.concatenate((pred_a, pred_b,
                                           np.expand_dims(horizon, 1),
                                           np.expand_dims(target, 1)), 1),
                           columns=["pred_a_{}".format(i) for i in range(num_ensemble)]
                                   + ["pred_b_{}".format(i) for i in range(num_ensemble)]
                                   + ['horizon', 'target'])
    ensemble_df, horizon_df = ensemble_voting(eval_df,
                                              ensemble_size_interval,
                                              num_ensemble,
                                              confidence_interval)
    assert (ensemble_df['accuracy'].values == 0).all()
    assert (horizon_df['accuracy'].values == 0).all()


@pytest.mark.parametrize('paired', [True, False])
def test_confidence_interval(paired):
    from core.uncertainty import confidence_interval
    import numpy as np
    import policybazaar
    import cque

    ensemble_size_interval = 5
    num_ensemble = 4
    env_name = 'Hopper-v2'

    queries = cque.get_queries(env_name)
    (policy_a_id, policy_b_id) = list(queries.keys())[0]
    query_batch = queries[(policy_a_id, policy_b_id)]
    policy_a, _ = policybazaar.get_policy(*policy_a_id)

    return_a = query_batch['info']['return_a']
    return_b = query_batch['info']['return_b']
    target = query_batch['target']
    horizon = query_batch['horizon']

    # #########################################################
    # accuracy should be 1 given ground truth estimates
    # #########################################################
    pred_a = np.repeat(np.expand_dims(np.array(return_a), 1),
                       num_ensemble, axis=1)
    pred_b = np.repeat(np.expand_dims(np.array(return_b), 1),
                       num_ensemble, axis=1)
    pred_a_col_names = ["pred_a_{}".format(i) for i in range(num_ensemble)]
    pred_b_col_names = ["pred_b_{}".format(i) for i in range(num_ensemble)]
    eval_df = pd.DataFrame(np.concatenate((pred_a, pred_b,
                                           np.expand_dims(horizon, 1),
                                           np.expand_dims(target, 1)), 1),
                           columns=(pred_a_col_names + pred_b_col_names
                                    + ['horizon', 'target']))
    ensemble_df, horizon_df = confidence_interval(eval_df,
                                                  ensemble_size_interval,
                                                  num_ensemble,
                                                  0.1,
                                                  paired=paired)
    assert (ensemble_df['accuracy'].values == 1).all()
    assert (horizon_df['accuracy'].values == 1).all()

    # #########################################################
    # accuracy should be 0 given swap of ground truth estimates
    # #########################################################
    pred_b = np.repeat(np.expand_dims(np.array(return_a), 1),
                       num_ensemble, axis=1)
    pred_a = np.repeat(np.expand_dims(np.array(return_b), 1),
                       num_ensemble, axis=1)

    eval_df = pd.DataFrame(np.concatenate((pred_a, pred_b,
                                           np.expand_dims(horizon, 1),
                                           np.expand_dims(target, 1)), 1),
                           columns=(pred_a_col_names
                                    + pred_b_col_names
                                    + ['horizon', 'target']))
    ensemble_df, horizon_df = confidence_interval(eval_df,
                                                  ensemble_size_interval,
                                                  num_ensemble,
                                                  0.1,
                                                  paired=paired)
    assert (ensemble_df['accuracy'].values[ensemble_df['abstain'].values
                                           != 1] == 0).all()
    assert (horizon_df['accuracy'].values[horizon_df['abstain'].values
                                          != 1] == 0).all()

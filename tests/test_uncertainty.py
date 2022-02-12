import pandas as pd
import pytest


def test_ensemble_voting():
    from core.uncertainty import ensemble_voting
    import numpy as np
    import opcc

    ensemble_size_interval = 5
    num_ensemble = 4
    env_name = 'Hopper-v2'

    queries = opcc.get_queries(env_name)
    (policy_a_id, policy_b_id) = list(queries.keys())[0]
    query_batch = queries[(policy_a_id, policy_b_id)]
    policy_a, _ = opcc.get_policy(*policy_a_id)

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
                                           np.expand_dims(target, 1),
                                           np.expand_dims(return_a, 1),
                                           np.expand_dims(return_b, 1)), 1),
                           columns=["pred_a_{}".format(i)
                                    for i in range(num_ensemble)]
                                   + ["pred_b_{}".format(i)
                                      for i in range(num_ensemble)]
                                   + ['horizon', 'target']
                                   + ['return_a', 'return_b'])
    uncertainty_dict = ensemble_voting(eval_df,
                                       ensemble_size_interval,
                                       num_ensemble)

    for ensemble_count in uncertainty_dict.keys():
        for h in uncertainty_dict[ensemble_count].keys():
            data = uncertainty_dict[ensemble_count][h]
            assert data['confidence'].mean() == 1
            assert (data['prediction'] == data['target']).all()

    # #########################################################
    # accuracy should be 0 given swap of ground truth estimates
    # #########################################################
    eval_df = pd.DataFrame(np.concatenate((pred_b, pred_a,
                                           np.expand_dims(horizon, 1),
                                           np.expand_dims(target, 1),
                                           np.expand_dims(return_a, 1),
                                           np.expand_dims(return_b, 1)), 1),
                           columns=["pred_a_{}".format(i)
                                    for i in range(num_ensemble)]
                                   + ["pred_b_{}".format(i)
                                      for i in range(num_ensemble)]
                                   + ['horizon', 'target']
                                   + ['return_a', 'return_b'])
    uncertainty_dict = ensemble_voting(eval_df,
                                       ensemble_size_interval,
                                       num_ensemble)

    for ensemble_count in uncertainty_dict.keys():
        for h in uncertainty_dict[ensemble_count].keys():
            data = uncertainty_dict[ensemble_count][h]
            assert data['confidence'].mean() == 1
            assert (data['prediction'] != data['target']).all()


@pytest.mark.parametrize('paired', [True, False])
def test_confidence_interval(paired):
    from core.uncertainty import confidence_interval
    import numpy as np
    import opcc

    ensemble_size_interval = 5
    num_ensemble = 4
    env_name = 'Hopper-v2'

    queries = opcc.get_queries(env_name)
    (policy_a_id, policy_b_id) = list(queries.keys())[0]
    query_batch = queries[(policy_a_id, policy_b_id)]
    policy_a, _ = opcc.get_policy(*policy_a_id)

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
                                           np.expand_dims(target, 1),
                                           np.expand_dims(return_a, 1),
                                           np.expand_dims(return_b, 1)), 1),
                           columns=["pred_a_{}".format(i)
                                    for i in range(num_ensemble)]
                                   + ["pred_b_{}".format(i)
                                      for i in range(num_ensemble)]
                                   + ['horizon', 'target']
                                   + ['return_a', 'return_b'])
    uncertainty_dict = confidence_interval(eval_df,
                                           ensemble_size_interval,
                                           num_ensemble,
                                           paired=paired)

    for ensemble_count in uncertainty_dict.keys():
        for h in uncertainty_dict[ensemble_count].keys():
            data = uncertainty_dict[ensemble_count][h]
            assert round(data['confidence'].mean(), 2) >= 0.99
            assert (data['prediction'] == data['target']).all()

    # #########################################################
    # accuracy should be 0 given swap of ground truth estimates
    # #########################################################

    eval_df = pd.DataFrame(np.concatenate((pred_b, pred_a,
                                           np.expand_dims(horizon, 1),
                                           np.expand_dims(target, 1),
                                           np.expand_dims(return_b, 1),
                                           np.expand_dims(return_a, 1)), 1),
                           columns=["pred_a_{}".format(i)
                                    for i in range(num_ensemble)]
                                   + ["pred_b_{}".format(i)
                                      for i in range(num_ensemble)]
                                   + ['horizon', 'target']
                                   + ['return_a', 'return_b'])
    uncertainty_dict = confidence_interval(eval_df,
                                           ensemble_size_interval,
                                           num_ensemble,
                                           paired=paired)

    for ensemble_count in uncertainty_dict.keys():
        for h in uncertainty_dict[ensemble_count].keys():
            data = uncertainty_dict[ensemble_count][h]
            assert round(data['confidence'].mean(), 2) >= 0.99
            assert (data['prediction'] != data['target']).all()

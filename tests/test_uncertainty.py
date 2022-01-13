import pandas as pd


def test_ensemble_voting():
    from core.uncertainty import ensemble_voting
    ensemble_size_interval = 5
    num_ensemble = 25
    confidence_interval = 0.1

    eval_df = pd.DataFrame()
    ensemble_df, uncertainty_df = ensemble_voting(eval_df,
                                                  ensemble_size_interval,
                                                  num_ensemble,
                                                  confidence_interval)

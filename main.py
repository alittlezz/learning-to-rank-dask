import dask
from dask.distributed import Client, LocalCluster, wait
from dask.dataframe import read_csv
from lightgbm import DaskLGBMRanker
from sklearn.model_selection import GroupKFold
import numpy as np
from lightgbm import DaskLGBMRanker, LGBMRanker
from copy import deepcopy
import warnings
import pandas as pd
import time
import argparse

warnings.filterwarnings('ignore')

NO_PARTITIONS = 10

def gbdt_ranking(X_train, y_train, qids_train, X_valid, y_valid, qids_valid, params):
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        verbose=-1,
        force_row_wise=True,
        **params
    )

    model.fit(
        X=X_train,
        y=y_train,
        group=qids_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[qids_valid],
        eval_at=[1, 5, 10],
        verbose=-1,
        early_stopping_rounds=int(params['n_estimators'] * 0.1)
    )
    return model.best_score_['valid_0']

def prepare_data(filename):
    train_df = pd.read_csv(filename)

    qids = train_df.groupby("QID")["QID"].count().to_numpy()
    X = train_df.drop(["QID", "Relevance"], axis=1)
    y = train_df["Relevance"]

    dx = X.to_numpy()
    dy = y.to_numpy()

    return dx, dy, qids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate the model using the best hyperparameters.')
    parser.add_argument('--model', default='gbdt', help='gbdt or rf (default gbdt)')
    args = parser.parse_args()
    if args.model == 'gbdt':
        params = {
            'learning_rate': 0.05,
            'num_leaves': 1000,
            'n_estimators': 2000,
        }
    elif args.model == 'rf':
        params = {
            'boosting_type': 'rf',
            'learning_rate': 0.05,
            'num_leaves': 1000,
            'n_estimators': 2000,
            'bagging_fraction': 0.2,
            'bagging_freq': 5
        }
    else:
        print('--model given', args.model, 'is not valid')
        exit(0)

    print('[1/2] Preparing data ...')

    X_train, y_train, qids_train = prepare_data('train.csv')
    X_valid, y_valid, qids_valid = prepare_data('test.csv')



    print('[2/2] Evaluating with hyperparamters ...', params)
    scores = gbdt_ranking(X_train, y_train, qids_train, X_valid, y_valid, qids_valid, params)
    print('Got scores', scores)
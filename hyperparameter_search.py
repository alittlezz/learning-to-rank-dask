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

warnings.filterwarnings('ignore')

NO_PARTITIONS = 10
qids = None
CV_FOLDS = 5
qids_train_dp = [None for _ in range(CV_FOLDS)]
qids_valid_dp = [None for _ in range(CV_FOLDS)]

USE_DASK = False


def partition_dataset(qids, X, y):
    N_queries = qids.size
    chunks_q = []
    chunks_data = []
    sm = 0
    for i in range(NO_PARTITIONS):
        if i == NO_PARTITIONS - 1:
            nr = N_queries - sum(chunks_q)
        else:
            nr = N_queries // NO_PARTITIONS
        chunks_q.append(nr)
        chunks_data.append(np.sum(qids[sm:sm + nr]))
        sm += nr

    chunks_data = dask.compute(chunks_data)[0]

    dx = X.to_dask_array(lengths=True)
    dy = y.to_dask_array(lengths=True)

    qids = qids.rechunk([tuple(chunks_q)])
    dx = dx.rechunk([tuple(chunks_data), (136,)])
    dy = dy.rechunk([tuple(chunks_data)])

    return dx, dy, qids

def rank_with_args(args):
    scores = dict.fromkeys(['ndcg@1', 'ndcg@5', 'ndcg@10'], 0)
    gkf = GroupKFold(n_splits=CV_FOLDS)
    i = 0
    st_it = time.time()
    for train_index, valid_index in gkf.split(dx, dy, groups):
        X_train, y_train = dx[train_index], dy[train_index]
        X_valid, y_valid = dx[valid_index], dy[valid_index]

        global qids_train_dp, qids_valid_dp
        if qids_train_dp[i] is None:
            print('Computing DP for', i)
            if USE_DASK:
                qids_train_dp[i] = dask.array.unique(groups[train_index]).compute()
                qids_valid_dp[i] = dask.array.unique(groups[valid_index]).compute()
            else:
                qids_train_dp[i] = np.unique(groups[train_index])
                qids_valid_dp[i] = np.unique(groups[valid_index])

        qids_train = qids[qids_train_dp[i]]
        qids_valid = qids[qids_valid_dp[i]]

        if USE_DASK:
            model = DaskLGBMRanker(
                client=client,
                objective="lambdarank",
                metric="ndcg",
                local_listen_port=8800,
                verbose=-1,
                **args
            )
        else:
            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                verbose=-1,
                force_row_wise=True,
                **args
            )

        model.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[(X_valid, y_valid)],
            eval_group=[qids_valid],
            eval_at=[1, 5, 10],
            verbose=-1,
            early_stopping_rounds=int(args['n_estimators'] * 0.1)
        )
        i += 1
        for key, value in model.best_score_['valid_0'].items():
            scores[key] += value


    for key in scores.keys():
        scores[key] /= CV_FOLDS

    return scores


def iterate_setting(setting, idx, params, best_scores):
    keys = list(setting.keys())
    if idx == len(keys):
        print('Running with params', params)
        scores = rank_with_args(params)
        print('Got scores', scores)
        for key in scores.keys():
            if scores[key] > best_scores[key][0]:
                best_scores[key] = (scores[key], deepcopy(params))
        return
    key = keys[idx]
    for value in setting[key]:
        params[key] = value
        iterate_setting(setting, idx + 1, params, best_scores)


if __name__ == "__main__":
    if USE_DASK:
        cluster = LocalCluster(n_workers=1)
        client = Client(cluster)
        print('[0/3] Created dashboard at', cluster.dashboard_link)

        train_df = read_csv('train.csv')

        print('[1/3] Finished data reading ...')

        qids = train_df.groupby("QID")["QID"].count().to_dask_array(lengths=True)
        X = train_df.drop(["QID", "Relevance"], axis=1)
        y = train_df["Relevance"]

        dx, dy, qids = partition_dataset(qids, X, y)
        groups = train_df['QID'].to_dask_array(lengths=True).rechunk(dy.chunks)
    else:
        train_df = pd.read_csv('train.csv')

        print('[1/3] Finished data reading ...')

        qids = train_df.groupby("QID")["QID"].count().to_numpy()
        X = train_df.drop(["QID", "Relevance"], axis=1)
        y = train_df["Relevance"]

        dx = X.to_numpy()
        dy = y.to_numpy()
        groups = train_df['QID'].to_numpy()

    print('[2/3] Finished creating train dataset ...')

    ranking_settings = [
        {
            'learning_rate': [0.01, 0.05],
            'num_leaves': [1000],
            'n_estimators': [2000],
            'boosting_type': ['rf'],
            'bagging_fraction': [0.2, 0.5, 0.8],
            'bagging_freq': [5, 20, 100]
        }
    ]

    best_scores = dict.fromkeys(['ndcg@1', 'ndcg@5', 'ndcg@10'], (0, None))

    print('[3/3] Finding best hyperparameters ...')
    for setting in ranking_settings:
        iterate_setting(setting, 0, {}, best_scores)

    print(best_scores)

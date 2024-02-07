import json
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from pandas.api.types import CategoricalDtype
import random
from utils import FEATURE_MAP, PROJECT_ROOT_DIR

CLASSIFIER_RESULTS_FILES_PATH = os.path.join(PROJECT_ROOT_DIR, "result")


def create_cv_folds(data, num_folds):
    transcript_files_sizes = data.transcript_file.value_counts()

    test_sets = [pd.DataFrame() for _ in range(num_folds)]

    while len(transcript_files_sizes) > 0:
        smallest_fold = np.argmin([len(s) for s in test_sets])
        largest_transcript = transcript_files_sizes.index[0]

        # Add the largest transcript to the smallest fold
        test_sets[smallest_fold] = pd.concat \
            ([test_sets[smallest_fold], data[data.transcript_file == largest_transcript]])
        del transcript_files_sizes[largest_transcript]

    train_sets = [data[~data.index.isin(data_test.index)].copy() for data_test in test_sets]

    print(f"Test set sizes: {[len(s) for s in test_sets]}")
    return train_sets, test_sets


def load_annotated_childes_data(path):
    transcripts = pd.read_json(path)
    return transcripts


def load_annotated_childes_datasplits(data_path, num_cv_folds=10, age=None):
    transcripts = load_annotated_childes_data(data_path)

    if age:
        transcripts = transcripts[transcripts["age"] == age ]

    print("Dataset size: ", len(transcripts))

    return create_cv_folds(transcripts, num_cv_folds)

def create_one_hot_encoded_dfs(train_df, test_df, col_name):
    unique_list = train_df[col_name].unique().tolist()
    unique_list.append("UNKOWN")
    train_df[col_name] = train_df[col_name].astype(CategoricalDtype(unique_list))
    test_df[col_name] = test_df[col_name].astype(CategoricalDtype(unique_list))
    dummies_train = pd.get_dummies(train_df[col_name], prefix='sa')
    dummies_test = pd.get_dummies(test_df[col_name], prefix='sa')
    train_df = pd.concat([train_df, dummies_train], axis=1).drop(col_name, axis=1)
    test_df = pd.concat([test_df, dummies_test], axis=1).drop(col_name, axis=1)
    return train_df, test_df

def train_classifier(data_path, num_cv, results_file, model):
    num_cv_folds = num_cv
    ages = [20, 32, None]
    results = []

    for age in ages:
        data_train_splits, data_test_splits = load_annotated_childes_datasplits(data_path, num_cv_folds, age)

        for k, v in FEATURE_MAP.items():
            print(f'Features:{k}')
            feature_cols = v
            weighted_f1_scores, mcc_scores = [], []

            for fold in range(num_cv_folds):
                if "majority" in feature_cols:
                    label_counts = data_train_splits[fold].annotated_score.value_counts()
                    largest_label = label_counts.index[0]
                    # print(f'label: {largest_label}')
                    y_test = data_test_splits[fold].annotated_score
                    y_pred = [largest_label] * len(y_test)

                elif "random" in feature_cols:
                    choice_set = [-1, 0, 1]
                    y_test = data_test_splits[fold].annotated_score
                    y_pred = random.choices(choice_set, k=len(y_test))

                else:
                    X_train = data_train_splits[fold].loc[:, feature_cols]
                    y_train = data_train_splits[fold].annotated_score
                    X_test = data_test_splits[fold].loc[:, feature_cols]
                    y_test = data_test_splits[fold].annotated_score
                    if 'concat_speech_acts' in feature_cols:
                        X_train, X_test = create_one_hot_encoded_dfs(X_train, X_test, 'concat_speech_acts')

                    if model == "random_forest":
                        clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
                    else:
                        clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=800).fit(X_train,
                                                                                                            y_train)
                    y_pred = clf.predict(X_test)

                # print(f'Fold #{fold}')

                # print(type(y_test))
                # print(type(y_pred))
                weighted_f1_score = f1_score(y_test, y_pred, average='weighted')
                weighted_f1_scores.append(weighted_f1_score)
                mcc_score = matthews_corrcoef(y_test, y_pred)
                mcc_scores.append(mcc_score)

            print(f'Mean weighted f1 score:{np.mean(weighted_f1_scores):.2f} Std: {np.std(weighted_f1_scores):.2f}')
            print(f'Mean MCC score:{np.mean(mcc_scores):.2f} Std: {np.std(mcc_scores):.2f}')
            result = {
                "features": k,
                "age": age,
                "model": "random_forest",
                "num_cv_folds": num_cv_folds,
                "mcc_mean": np.mean(mcc_scores),
                "mcc_std": np.std(mcc_scores),
                "mcc_scores": mcc_scores,
                "weighted_f1_mean": np.mean(weighted_f1_scores),
                "weighted_f1_std": np.std(weighted_f1_scores),
                "weighted_f1_scores": weighted_f1_scores
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    temp_df_1 = pd.DataFrame(results_df['mcc_scores'].to_list(),
                             columns=['mcc_cv_1', 'mcc_cv_2', 'mcc_cv_3', 'mcc_cv_4', 'mcc_cv_5'])
    temp_df_2 = pd.DataFrame(results_df['weighted_f1_scores'].to_list(),
                             columns=['f1_cv_1', 'f1_cv_2', 'f1_cv_3', 'f1_cv_4', 'f1_cv_5'])

    df = pd.concat([results_df, temp_df_1, temp_df_2], axis=1).drop(['mcc_scores', 'weighted_f1_scores'], axis=1)

    df.to_csv(CLASSIFIER_RESULTS_FILES_PATH + results_file)

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-path",
        type=str,
        help="json file path to data"
    )
    argparser.add_argument(
        "--model",
        type=str,
        help="classifier model type 'regression' or 'random_forest'",
        default="regression"
    )
    argparser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cv folds"
    )
    argparser.add_argument(
        "--results-path",
        type=str,
        help="file path to store results"
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    train_classifier(args.data_path, args.cv_folds, args.results_path, args.model)
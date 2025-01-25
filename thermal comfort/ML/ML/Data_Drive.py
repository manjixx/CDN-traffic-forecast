# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


def train():
    """Soft-SVM"""
    svc = SVC(kernel='rbf', decision_function_shape='ovr')
    svc.fit(x_train, y_train)
    joblib.dump(svc, f'./Data/soft_svm.pkl')

    """KNN"""
    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None
    )

    knn.fit(x_train, y_train)
    joblib.dump(knn, f'./Data/knn.pkl')

    """Random Forest"""
    rf = RandomForestClassifier(
        max_depth=15,
        random_state=2019
    )
    rf.fit(x_train, y_train)
    joblib.dump(rf, f'./Data/random_forest.pkl')

    """Xgboost"""
    xgb = LGBMClassifier(
        learning_rate=0.0075,
        max_depth=15,
        n_estimators=2048,
        num_leaves=63,
        random_state=2019,
        n_jobs=-1,
        reg_alpha=0.8,
        reg_lambda=0.8,
        subsample=0.2,
        colsample_bytree=0.5,
    )
    xgb.fit(x_train, y_train)
    joblib.dump(rf, f'./Data/xgboost.pkl')


def evaluate(y_pred, y_test):
    print('准确率：' + str(accuracy_score(y_pred, y_test)))
    print('精确率 macro：' + str(precision_score(y_pred, y_test, average='macro')))
    print('精确率 micro：' + str(precision_score(y_pred, y_test, average='micro')))
    print('精确率 weighted：' + str(precision_score(y_pred, y_test, average='weighted')))
    print('Recall macro：' + str(recall_score(y_pred, y_test, average='macro')))
    print('Recall micro：' + str(recall_score(y_pred, y_test, average='micro')))
    print('Recall weighted：' + str(recall_score(y_pred, y_test, average='weighted')))
    print('F1-score macro：' + str(f1_score(y_pred, y_test, average='macro')))
    print('F1-score micro：' + str(f1_score(y_pred, y_test, average='micro')))
    print('F1-score weighted：' + str(f1_score(y_pred, y_test, average='weighted')))


def test(x_test, y_test):

    model = ['soft_svm', 'knn', 'random_forest', 'xgboost']
    for m in model:
        print("**" * 10)
        print(m)
        print("**"*10)
        svc = joblib.load(f'./Data/{m}.pkl')
        y_pred = svc.predict(x_test)
        evaluate(y_pred, y_test)



def data_load():
    df = pd.read_csv('../dataset/format/dataset.csv').dropna(axis=0, how='any', inplace=False)
    normalization = ['count', 'age', 'height', 'weight', 'ta', 'hr', 'bmi']
    other = ['gender', 'season', 'griffith']
    y_feature = 'tsv'
    if category == 2:
        # df = df.drop(df[df[y_feature] == 1].index, inplace=False)
        df.loc[(df[y_feature] != 1), y_feature] = 0

    normalization = df[normalization].reset_index(drop=True)
    other = df[other].reset_index(drop=True)
    y = df[y_feature].reset_index(drop=True)
    normalization = scaler.fit_transform(normalization)
    x = np.concatenate([normalization, other], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    category = 3
    scaler = MinMaxScaler()
    x_train, x_test, y_train, y_test = data_load()
    train()
    test(x_test, y_test)



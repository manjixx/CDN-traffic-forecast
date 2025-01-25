# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism
@File ：xgboost.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/6 20:33
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

def dataloader():
    filename = 'synthetic'
    env = np.load(f'../dataset/{filename}/env.npy', allow_pickle=True).astype(float)  # ta hr va
    season = np.load(f'../dataset/{filename}/season.npy', allow_pickle=True).astype(int)  # season
    date = np.load(f'../dataset/{filename}/date.npy', allow_pickle=True)  # date
    body = np.load(f'../dataset/{filename}/body.npy', allow_pickle=True).astype(float)  # age height weight bmi
    # griffith, gender, sensitivity, preference, environment
    gender = np.load(f'../dataset/{filename}/gender.npy', allow_pickle=True)
    # gender = gender[:, 0:2]
    y = np.load(f'../dataset/{filename}/label.npy', allow_pickle=True).astype(int)  # pmv

    # normalization: [ta hr va age height weight bmi]
    x = np.concatenate((env, body), axis=1)
    x = MinMaxScaler().fit_transform(x)
    # season ta hr va age height weight bmi griffith, gender pmv
    x = np.concatenate((season, gender, x), axis=1)

    train_feature, test_feature, train_label, test_label = train_test_split(x, y, test_size=0.2)
    print(f'train_feature shape: {len(train_feature)} * {len(train_feature[0])}')
    print(f'test_feature shape: {len(test_feature)} * {len(test_feature[0])}')

    return np.array(train_feature), np.array(test_feature), np.array(train_label), np.array(
        test_label)


if __name__ == '__main__':
    train_feature, test_feature, train_label, test_label = dataloader()
    print('*********soft svm*************')
    model = SVC(kernel='linear', decision_function_shape='ovo', C=1).fit(train_feature, train_label)
    model.fit(train_feature, train_label)
    y_pred = model.predict(test_feature)
    print('准确率：' + str(accuracy_score(y_pred, test_label)))
    print('精确率 macro：' + str(precision_score(y_pred, test_label, average='macro')))
    print('精确率 micro：' + str(precision_score(y_pred, test_label, average='micro')))
    print('精确率 weighted：' + str(precision_score(y_pred, test_label, average='weighted')))
    print('Recall macro：' + str(recall_score(y_pred, test_label, average='macro')))
    print('Recall micro：' + str(recall_score(y_pred, test_label, average='micro')))
    print('Recall weighted：' + str(recall_score(y_pred, test_label, average='weighted')))
    print('F1-score macro：' + str(f1_score(y_pred, test_label, average='macro')))
    print('F1-score micro：' + str(f1_score(y_pred, test_label, average='micro')))
    print('F1-score weighted：' + str(f1_score(y_pred, test_label, average='weighted')))

    print('*********knn manhattan*************')

    model = KNeighborsClassifier(
        n_neighbors=9,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        n_jobs=None
    )
    model.fit(train_feature, train_label)
    y_pred = model.predict(test_feature)
    print('准确率：' + str(accuracy_score(y_pred, test_label)))
    print('精确率 macro：' + str(precision_score(y_pred, test_label, average='macro')))
    print('精确率 micro：' + str(precision_score(y_pred, test_label, average='micro')))
    print('精确率 weighted：' + str(precision_score(y_pred, test_label, average='weighted')))
    print('Recall macro：' + str(recall_score(y_pred, test_label, average='macro')))
    print('Recall micro：' + str(recall_score(y_pred, test_label, average='micro')))
    print('Recall weighted：' + str(recall_score(y_pred, test_label, average='weighted')))
    print('F1-score macro：' + str(f1_score(y_pred, test_label, average='macro')))
    print('F1-score micro：' + str(f1_score(y_pred, test_label, average='micro')))
    print('F1-score weighted：' + str(f1_score(y_pred, test_label, average='weighted')))

    print('*********Random-Forest*************')

    classifier = RandomForestClassifier(
        max_depth=5,
        random_state=2019,
        n_estimators=30
    )

    classifier.fit(train_feature, train_label)
    y_pred = classifier.predict(test_feature)
    print('准确率：' + str(accuracy_score(y_pred, test_label)))
    print('精确率 macro：' + str(precision_score(y_pred, test_label, average='macro')))
    print('精确率 micro：' + str(precision_score(y_pred, test_label, average='micro')))
    print('精确率 weighted：' + str(precision_score(y_pred, test_label, average='weighted')))
    print('Recall macro：' + str(recall_score(y_pred, test_label, average='macro')))
    print('Recall micro：' + str(recall_score(y_pred, test_label, average='micro')))
    print('Recall weighted：' + str(recall_score(y_pred, test_label, average='weighted')))
    print('F1-score macro：' + str(f1_score(y_pred, test_label, average='macro')))
    print('F1-score micro：' + str(f1_score(y_pred, test_label, average='micro')))
    print('F1-score weighted：' + str(f1_score(y_pred, test_label, average='weighted')))

    print('*********Xgboost*************')

    classifier = LGBMClassifier(
        learning_rate=0.008,
        max_depth=5,
        n_estimators=90,
        num_leaves=63,
        random_state=2019,
        n_jobs=-1,
        reg_alpha=0.8,
        reg_lambda=0.8,
        subsample=0.2,
        colsample_bytree=0.5,
    )
    classifier.fit(train_feature, train_label)
    y_pred = classifier.predict(test_feature)
    print('准确率：' + str(accuracy_score(y_pred, test_label)))
    print('精确率 macro：' + str(precision_score(y_pred, test_label, average='macro')))
    print('精确率 micro：' + str(precision_score(y_pred, test_label, average='micro')))
    print('精确率 weighted：' + str(precision_score(y_pred, test_label, average='weighted')))
    print('Recall macro：' + str(recall_score(y_pred, test_label, average='macro')))
    print('Recall micro：' + str(recall_score(y_pred, test_label, average='micro')))
    print('Recall weighted：' + str(recall_score(y_pred, test_label, average='weighted')))
    print('F1-score macro：' + str(f1_score(y_pred, test_label, average='macro')))
    print('F1-score micro：' + str(f1_score(y_pred, test_label, average='micro')))
    print('F1-score weighted：' + str(f1_score(y_pred, test_label, average='weighted')))

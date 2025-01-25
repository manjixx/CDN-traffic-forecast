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
import tensorflow as tf


def data_loader():
    filepath = 'Synthetic'

    """
    'male', 'female','young', 'old','short', 'medium', 'tall','thin', 'normal', 'fat',
    'bmi_l', 'bmi_n', 'bmi_h',  'grf_l', 'grf_n', 'grf_h','sen_l', 'sen_n', 'sen_h',
    'pre_l', 'pre_n', 'pre_h', 'env_l', 'env_n', 'env_h'
    """
    person = np.load('../Dataset/npy/' + filepath + '/person.npy', allow_pickle=True).astype(float)
    print(len(person))
    # 'date', 'time', 'season', 'va', 'ta', 'hr'
    env = np.load('../Dataset/npy/' + filepath + '/env.npy', allow_pickle=True)
    ta = env[:, 3:6]
    season = env[:, 2:3]

    # ta_diff1, ta_diff2
    diff = np.load('../Dataset/npy/' + filepath + '/diff.npy', allow_pickle=True).astype(float)
    # 'age_avg', 'height_avg', 'weight_avg', 'bmi_avg'
    avg = np.load('../Dataset/npy/' + filepath + '/avg.npy', allow_pickle=True).astype(float)
    # griffith_avg
    griffith = np.load('../Dataset/npy/' + filepath + '/grf.npy', allow_pickle=True).astype(float)
    # count
    count = np.load('../Dataset/npy/' + filepath + '/count.npy', allow_pickle=True)
    # label
    y = np.load('../Dataset/npy/' + filepath + '/tsv.npy', allow_pickle=True).astype(float)[:, None]

    # normalization: ['va', 'ta', 'hr', 'height_avg', 'weight_avg', 'bmi_avg'] env, avg
    normalization = np.concatenate((ta, avg), axis=1)
    normalization = MinMaxScaler().fit_transform(X=normalization)

    # count, person, griffith_avg, season, diff, va, ta, hr, age_avg, height_avg, weight_avg, bmi_avg

    x = np.concatenate((count[:, None], person, griffith[:, None], season, diff, normalization), axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print(f'x_train shape: {np.array(x_train).shape}')
    print(f'y_train shape: {np.array(y_train).shape}')
    print(f'x_test shape: {np.array(x_test).shape}')
    print(f'y_test shape: {np.array(y_test).shape}')

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    train_feature, test_feature, train_label, test_label = data_loader()
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

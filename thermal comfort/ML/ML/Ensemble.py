# -*- coding: utf-8 -*-
import pandas as pd

from weight import *
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def data_loader():
    file_path = "../dataset/data/dataset.csv"

    df = pd.read_csv(file_path).dropna(axis=0, how='any', inplace=False)
    df.rename(columns={'thermal sensation': 'thermal_sensation'}, inplace=True)

    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0
    # df = df.drop(df.index[(df.thermal_sensation == 1)])


    w1, w2, w3 = weight_griffith(df)
    w_max = max(w1, w2, w3)
    w1, w2, w3 = w1 / w_max, w2 / w_max, w3 / w_max
    w_g = []

    for g in np.array(df['griffith']):
        if g <= g_down:
            w_g.append(w1)
        elif g >= g_upper:
            w_g.append(w3)
        else:
            w_g.append(w2)
    w_g = pd.DataFrame({'w': w_g})

    w1, w2, w3 = weight_bmi(df)
    w_max = max(w1, w2, w3)
    w1, w2, w3 = w1 / w_max, w2 / w_max, w3 / w_max

    w_b = []

    for b in np.array(df['bmi']):
        if b <= b_down:
            w_b.append(w1)
        elif b >= b_upper:
            w_b.append(w3)
        else:
            w_b.append(w2)
    w_b = pd.DataFrame({'w': w_b})

    normal = df[['age', 'height', 'weight', 'ta', 'hr', 'bmi']]
    x = scaler.fit_transform(X=normal)
    other = df[['gender', 'season', 'griffith']]
    x = np.concatenate([x, other], axis=1)
    print(x.shape)

    y = df[y_feature]

    return x, y, w_b, w_g


def train_test(x, y, weight):
    if weight.shape[0] != 0:
        x = np.concatenate([x, weight], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    return x_train, x_test, y_train, y_test


def evaluate(y_test, y_pred, sample_weight):
    print('准确率：' + str(accuracy_score(y_pred, y_test, sample_weight=sample_weight)))
    print('精确率 macro：' + str(precision_score(y_pred, y_test, average='macro', sample_weight=sample_weight)))
    print('精确率 micro：' + str(precision_score(y_pred, y_test, average='micro', sample_weight=sample_weight)))
    print('精确率 weighted：' + str(precision_score(y_pred, y_test, average='weighted', sample_weight=sample_weight)))
    print('Recall macro：' + str(recall_score(y_pred, y_test, average='macro', sample_weight=sample_weight)))
    print('Recall micro：' + str(recall_score(y_pred, y_test, average='micro', sample_weight=sample_weight)))
    print('Recall weighted：' + str(recall_score(y_pred, y_test, average='weighted', sample_weight=sample_weight)))
    print('F1-score macro：' + str(f1_score(y_pred, y_test, average='macro', sample_weight=sample_weight)))
    print('F1-score micro：' + str(f1_score(y_pred, y_test, average='micro', sample_weight=sample_weight)))
    print('F1-score weighted：' + str(f1_score(y_pred, y_test, average='weighted', sample_weight=sample_weight)))


def data_evaluate(y_test, y_pred):
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


def data_driven(x, y):
    print('\n数据驱动模型 RandomForest')
    weight = pd.DataFrame({'w': []})
    x_train, x_test, y_train, y_test = train_test(x, y, weight)

    classifier = RandomForestClassifier(
        max_depth=3, random_state=2019
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    data_evaluate(y_test, y_pred)
    print('\n数据驱动模型 Xgboost')
    classifier = LGBMClassifier(
        learning_rate=0.0075,
        max_depth=3,
        n_estimators=2048,
        num_leaves=63,
        random_state=2019,
        n_jobs=-1,
        reg_alpha=0.8,
        reg_lambda=0.8,
        subsample=0.2,
        colsample_bytree=0.5,
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    data_evaluate(y_test, y_pred)


def data_phy(x, y, weight):
    print('\n数据机理双驱动 RandomForest')
    x_train, x_test, y_train, y_test = train_test(x, y, weight)

    classifier = RandomForestClassifier(
        max_depth=7, random_state=2019
    )

    classifier.fit(x_train[:, 0:9], y_train, sample_weight=x_train[:, 9:].flatten())
    y_pred = classifier.predict(x_test[:, 0:9])
    evaluate(y_test, y_pred, sample_weight=x_test[:, 9:].flatten())
    print('\n数据机理双驱动 Xgboost')

    classifier = LGBMClassifier(
        learning_rate=0.0075,
        max_depth=7,
        n_estimators=2048,
        num_leaves=63,
        random_state=2019,
        n_jobs=-1,
        reg_alpha=0.8,
        reg_lambda=0.8,
        subsample=0.2,
        colsample_bytree=0.5,
    )
    classifier.fit(x_train[:, 0:9], y_train, sample_weight=x_train[:, 9:].flatten())
    y_pred = classifier.predict(x_test[:, 0:9])
    evaluate(y_test, y_pred, sample_weight=x_test[:, 9:].flatten())


if __name__ == '__main__':

    x_feature = ['gender', 'age', 'height', 'weight', 'ta', 'hr', 'season', 'bmi', 'griffith']
    y_feature = 'thermal_sensation'

    alpha, beta = 0.5, 0.5
    b_down, b_upper = 18.5, 25
    g_down, g_upper = 1.0, 2.0
    scaler = MinMaxScaler()

    x, y, bmi_weight, griffith_weight = data_loader()

    data_driven(x, y)

    print("\n权重系数为bmi: ")
    data_phy(x, y, bmi_weight)

    print("\n 权重系数为grffith: ")
    data_phy(x, y, griffith_weight)








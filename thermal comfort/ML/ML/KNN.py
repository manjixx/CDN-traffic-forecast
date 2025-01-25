# -*- coding: utf-8 -*-
import pandas as pd

from weight import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def data_loader():
    file_path = "../dataset/data/dataset.csv"

    df = pd.read_csv(file_path).dropna(axis=0, how='any', inplace=False)
    df.rename(columns={'thermal sensation': 'thermal_sensation'}, inplace=True)

    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0
    # df = df.drop(df.index[(df.thermal_sensation == 1)])

    w1, w2, w3 = weight_griffith(df)
    w_max = min(w1, w2, w3)
    w1, w2, w3 = w1 / w_max, w2 / w_max, w3 / w_max

    print(w1, w2, w3)
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
    w_max = min(w1, w2, w3)
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
    x_b = np.concatenate([x, other, w_b], axis=1)
    x_g = np.concatenate([x, other, w_g], axis=1)

    y = df[y_feature]

    return x_b, x_g, y


def train_test(x, y, weight):
    if weight.shape[0] != 0:
        x = np.concatenate([x, weight], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


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
    print('\n数据驱动模型 KNN')
    x = x[:, 0:9]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    classifier = KNeighborsClassifier(
        n_neighbors=21,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None
    )

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    data_evaluate(y_test, y_pred)


def caller(args, func):
    func(args)


def weight(x):
    print(x.shape)
    return x[:, 9:10].flatten()


def data_phy(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = KNeighborsClassifier(
        n_neighbors=9,
        weights=caller(x_train, weight),
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None
    )
    x_train = x_train[:, 0:9]
    x_test = x_test[:, 0:9]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    data_evaluate(y_test, y_pred)





if __name__ == '__main__':

    x_feature = ['gender', 'age', 'height', 'weight', 'ta', 'hr', 'season', 'bmi', 'griffith']
    y_feature = 'thermal_sensation'

    alpha, beta = 0.5, 0.5
    b_down, b_upper = 18.5, 25
    g_down, g_upper = 1.0, 2.0
    scaler = MinMaxScaler()

    x_b, x_g, y = data_loader()

    print('\n 数据驱动算法')
    data_driven(x_b, y)

    print("\n权重系数为bmi: ")
    data_phy(x_b, y)

    print("\n 权重系数为grffith: ")
    data_phy(x_g, y)








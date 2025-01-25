# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from random import sample
import warnings
warnings.filterwarnings("ignore")


def split(data, index):
    if index == 'bmi':
        upper = b_up
        down = b_down
    else:
        upper = g_up
        down = g_down

    data1 = data[(data[index] <= down)].reset_index(drop=True)
    data2 = data[(data[index] > down) & (data[index] < upper)].reset_index(drop=True)
    data3 = data[(data[index] >= upper)].reset_index(drop=True)

    return data1, data2, data3


def weight_bmi(df):

    no = df['no'].unique()
    level1, level2, level3 = 0, 0, 0

    for n in no:
        data = df.loc[(df['no'] == n)]
        b = data['bmi'].unique()[0]

        if b <= b_down:
            level1 += 1
        elif b >= b_up:
            level3 += 1
        else:
            level2 += 1
    total = no.shape[0]
    p1 = level1 / total
    p2 = level2 / total
    p3 = level3 / total

    bmi1, bmi2, bmi3 = split(df, index='bmi')

    p4 = bmi1[(bmi1[y_feature] != 1)].shape[0] / bmi1.shape[0]
    p5 = bmi2[(bmi2[y_feature] != 1)].shape[0] / bmi2.shape[0]
    p6 = bmi3[(bmi3[y_feature] != 1)].shape[0] / bmi3.shape[0]

    w1 = alpha * p1 + beta * p4
    w2 = alpha * p2 + beta * p5
    w3 = alpha * p3 + beta * p6

    return round(w1, 2), round(w2, 2), round(w3, 2)


def weight_griffith(df):

    no = df['no'].unique()
    level1, level2, level3 = 0, 0, 0

    for n in no:
        data = df.loc[(df['no'] == n)]
        g = data['griffith'].unique()[0]
        if g <= g_down:
            level1 += 1
        elif g >= g_up:
            level3 += 1
        else:
            level2 += 1

    total = no.shape[0]
    p1 = level1 / total
    p2 = level2 / total
    p3 = level3 / total

    griffith1, griffith2, griffith3 = split(df, 'griffith')

    p4 = griffith1[(griffith1[y_feature] != 1)].shape[0] / griffith1.shape[0]
    p5 = griffith2[(griffith2[y_feature] != 1)].shape[0] / griffith2.shape[0]
    p6 = griffith3[(griffith3[y_feature] != 1)].shape[0] / griffith3.shape[0]

    w1 = alpha * p1 + beta * p4
    w2 = alpha * p2 + beta * p5
    w3 = alpha * p3 + beta * p6
    return round(w1, 2), round(w2, 2), round(w3, 2)


def weight(df, origin):
    no = df['no'].unique().tolist()
    no = sorted(sample(no, 45))
    origin = origin[origin['no'].isin(no)]
    no = origin['no'].unique().tolist()

    if index == 'bmi':
        w1, w2, w3 = weight_bmi(origin)
    else:
        w1, w2, w3 = weight_griffith(origin)
    return w1, w2, w3


def data_load():
    df = pd.read_csv('./predict.csv').dropna(axis=0, how='any', inplace=False)
    orgDf = pd.read_csv('../../dataset/format/dataset.csv')
    return df, orgDf


def cal(data):
    data1, data2, data3 = split(data, index)
    p1 = data1[(data1[y_feature] != 1)].shape[0] / data1.shape[0]
    p2 = data2[(data2[y_feature] != 1)].shape[0] / data2.shape[0]
    p3 = data3[(data3[y_feature] != 1)].shape[0] / data3.shape[0]

    p = w1 * p1 + w2 * p2 + w3 * p3

    return p


def output(df):
    env = df.drop_duplicates(subset=['ta', 'hr'])[['ta', 'hr']]
    res = []
    for e in np.array(env):
        r = []
        ta = e[0]
        hr = e[1]
        r.append(ta)
        r.append(hr)
        data = df[(df['ta'] == ta) & (df['hr'] == hr)]
        p = cal(data)
        # print(ta, hr, p)
        if p >= 0.2:
            # 0 不舒适
            r.append(0)
        else:
            r.append(1)
        res.append(r)
    res = pd.DataFrame(columns=['ta', 'hr', 'tsv'], data=res)
    return res


if __name__ == '__main__':
    b_up, b_down = 25, 18.5
    g_up, g_down = 2.0, 1.0
    alpha, beta = 0.5, 0.5
    for i in ['griffith', 'bmi']:
        index = i
        y_feature = 'tsv'
        predict, origin = data_load()
        w1, w2, w3 = weight(predict, origin)
        w = [w1, w2, w3]
        res = output(predict)
        res.to_csv(f'../PLOT/dataset/{index}.csv', index=False)

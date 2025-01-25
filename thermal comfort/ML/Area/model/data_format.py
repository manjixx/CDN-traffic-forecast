# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from random import sample
import warnings
warnings.filterwarnings("ignore")


def data_load():
    df = pd.read_csv('./data_predict.csv').dropna(axis=0, how='any', inplace=False)
    return df


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
        p = data[(data[y_feature] != 1)].shape[0] / data.shape[0]
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
    y_feature = 'tsv'
    predict = data_load()
    res = output(predict)
    res.to_csv(f'../PLOT/dataset/data.csv', index=False)


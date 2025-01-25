# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


def test():
    svc = joblib.load(f'../../ML/Data/random_forest.pkl')
    y_pred = svc.predict(x_test)
    print(y_pred)
    return y_pred


def data_load():
    df = pd.read_csv('../dataset/synthetic.csv').dropna(axis=0, how='any', inplace=False)

    normalization = ['count', 'age', 'height', 'weight', 'ta', 'hr', 'bmi']
    other = ['gender', 'season', 'griffith']

    normalization = df[normalization].reset_index(drop=True)
    other = df[other].reset_index(drop=True)
    normalization = scaler.fit_transform(normalization)
    x = np.concatenate([normalization, other], axis=1)
    print(f'x_test shape{x.shape}')
    return df, x


if __name__ == '__main__':

    category = 3
    scaler = MinMaxScaler()
    df, x_test = data_load()
    y_pred = test()

    tsv = pd.DataFrame(columns=['tsv'], data=y_pred)
    person = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith', 'ta', 'hr']
    person = df[person]
    df = pd.concat([person, tsv], axis=1)
    df.to_csv('./data_predict.csv', index=False)



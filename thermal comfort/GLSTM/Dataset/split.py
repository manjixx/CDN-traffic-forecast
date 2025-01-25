# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism
@File ：csv2npy_Ver1.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/5 20:18
"""

import pandas as pd
import os
import shutil
import numpy as np
import random
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from random import sample
import seaborn as sns


def load():
    df = pd.read_csv(file_path + '2021.csv').dropna(axis=0, how='any', inplace=False)
    df.loc[(df['time'] == '9:00:00'), 'time'] = '09:00:00'
    df.loc[(df['time'] == '9:30:00'), 'time'] = '09:30:00'
    df.loc[(df['date'] == '2021/1/9'), 'date'] = '2021/1/09'
    df.loc[(df['season'] == 'summer'), 'season'] = 0
    print(f'夏季数据条数{df.loc[(df["season"] == 0)].shape[0]}')
    df.loc[(df['season'] == 'winter'), 'season'] = 1
    print(f'冬季数据条数{df.loc[(df["season"] == 1)].shape[0]}')

    df.rename(columns={'thermal sensation': 'tsv'}, inplace=True)
    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0
    df = df.sort_values(by=['no', 'date', 'time'], axis=0, ascending=True, inplace=False).reset_index(drop=True)

    data = df.drop(df.index[(df.time > '17:30:00') | (df.time >= '12:30:00') & (df.time <= '14:00:00')])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 20) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 56) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 25) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/29') & (data.no == 33) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/25') & (data.no == 49) & (data.time == '12:00:00')])

    return data


def load_syn():
    df = pd.read_csv(file_path + 'synthetic.csv').dropna(axis=0, how='any', inplace=False)
    df.loc[(df['time'] == '9:00:00'), 'time'] = '09:00:00'
    df.loc[(df['time'] == '9:30:00'), 'time'] = '09:30:00'
    df.loc[(df['date'] == '2021/1/9'), 'date'] = '2021/1/09'
    df.loc[(df['season'] == 'summer'), 'season'] = 0
    print(f'夏季数据条数{df.loc[(df["season"] == 0)].shape[0]}')
    df.loc[(df['season'] == 'winter'), 'season'] = 1
    print(f'冬季数据条数{df.loc[(df["season"] == 1)].shape[0]}')

    df.rename(columns={'thermal sensation': 'tsv'}, inplace=True)
    data = df.sort_values(by=['no', 'date', 'time'], axis=0, ascending=True, inplace=False).reset_index(drop=True)
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0

    return data


def output(df):
    date = df['date'].sort_values().unique()
    for i in range(6, 13):
        print(f'{i}人数据集')
        path = './' + save_path + f'/{i}/'
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path, ignore_errors=True)
        for d in date:
            data = df.loc[df['date'] == d].reset_index(drop=True)
            no = data['no'].unique().tolist()
            if len(no) < i:
                break
            if len(no) - i >= 3:
                if data['season'].unique()[0] == 1:
                    up = 4
                else:
                    up = 4
                for j in range(0, up):
                    no = sorted(sample(no, i))
                    data = data[data['no'].isin(no)]
                    data = data.sort_values(by=['no', 'time'], axis=0, ascending=True, inplace=False).reset_index(
                        drop=True)
                    name = path + d.split('/')[1] + '-' + d.split('/')[2] + '-' + str(j) + '.csv'

                    data.to_csv(name, index=False)
            else:
                no = sorted(sample(no, i))
                data = data[data['no'].isin(no)].reset_index(drop=True)
                name = path + d.split('/')[1] + '-' + d.split('/')[2] + '.csv'
                data.to_csv(name, index=False)


if __name__ == '__main__':

    file_path = '../../DataSet/'
    y_feature = 'tsv'
    save_path = 'ORGDATA'
    df = load()
    # save_path = 'Synthetic'
    # df = load_syn()
    output(df)







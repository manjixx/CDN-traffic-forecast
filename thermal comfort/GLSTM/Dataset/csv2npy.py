# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism
@File ：csv2npy_Ver1.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/5 20:18
"""

import pandas as pd
import numpy as np
import random
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

if __name__ == '__main__':

    name = ['male', 'female',
            'young', 'old',
            'short', 'medium', 'tall',
            'thin', 'normal', 'fat',
            'bmi_l', 'bmi_n', 'bmi_h',
            'grf_l', 'grf_n', 'grf_h',
            'sen_l', 'sen_n', 'sen_h',
            'pre_l', 'pre_n', 'pre_h',
            'env_l', 'env_n', 'env_h',
            'date', 'time', 'ta', 'hr', 'season', 'va',
            'height_avg', 'weight_avg', 'bmi_avg', 'griffith_avg',
            'tsv']

    # 性别特征不需要归一化最后添加
    person_feature = ['male', 'female',
                      'young', 'old',
                      'short', 'medium', 'tall',
                      'thin', 'normal', 'fat',
                      'bmi_l', 'bmi_n', 'bmi_h',
                      'grf_l', 'grf_n', 'grf_h',
                      'sen_l', 'sen_n', 'sen_h',
                      'pre_l', 'pre_n', 'pre_h',
                      'env_l', 'env_n', 'env_h']
    env_feature = ['date', 'time', 'season', 'va', 'ta', 'hr']
    diff_feature = ['ta_diff1', 'ta_diff2']
    avg_feature = ['age_avg', 'height_avg', 'weight_avg', 'bmi_avg']
    griffith = 'griffith_avg'
    count_feature = 'count'
    y_feature = 'tsv'

    file = ['./synthetic.csv', './dataset.csv']
    for f in file:

        df = pd.read_csv(f).dropna(axis=0, how='any', inplace=False)

        # 房间人数
        count = df[count_feature].reset_index(drop=True)
        # 人员特征
        person = df[person_feature].reset_index(drop=True)
        # 环境数据
        env = df[env_feature].reset_index(drop=True)
        # 平均值
        avg = df[avg_feature].reset_index(drop=True)
        # 格里菲斯常数
        grf = df[griffith].reset_index(drop=True)
        # diff
        diff = df[diff_feature].reset_index(drop=True)
        # 标签
        tsv = df[y_feature].reset_index(drop=True)

        print(count, person, env, avg, grf, diff, tsv)
        '''save data'''

        if f == './synthetic.csv':
            save = 'synthetic'
        else:
            save = 'dataset'
        np.save('./npy/' + save + '/count.npy', count)
        np.save('./npy/' + save + '/person.npy', person)
        np.save('./npy/' + save + '/env.npy', env)
        np.save('./npy/' + save + '/avg.npy', avg)
        np.save('./npy/' + save + '/grf.npy', grf)
        np.save('./npy/' + save + '/diff.npy', diff)
        np.save('./npy/' + save + '/tsv.npy', tsv)

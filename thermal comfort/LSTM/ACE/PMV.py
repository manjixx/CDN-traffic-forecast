# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：PMV.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/4/1 16:46 
"""
from PMV_Model import *
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../../dataset/synthetic.csv').dropna(axis=0, how='any', inplace=False)
    df.loc[(df['time'] == '9:00:00'), 'time'] = '09:00:00'
    df.loc[(df['time'] == '9:30:00'), 'time'] = '09:30:00'
    data = df.sort_values(by=['no', 'date', 'time'], axis=0, ascending=True, inplace=False).reset_index(drop=True)
    env_feature = ['ta', 'hr', 'va']
    env = data[env_feature].reset_index(drop=True)
    # 季节
    season = []
    for i in np.array(data[['season']]).flatten():
        if i == 'summer':
            season.append(0)
        elif i == 'winter':
            season.append(1)
    season = pd.DataFrame({'season': season})
    x = np.concatenate((env, season), axis=1)
    y = data['thermal sensation'].values
    y = np.array(y)
    print("算法：pmv")
    print(x)
    print(y)
    pmv(x, y)
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


def deal_2018():

    # 数据清洗
    df = pd.read_csv('../../dataset/2018.csv').dropna(axis=0, how='any', inplace=False)
    data = df[(df.time != '8:50:00') & (df.time != '14:20:00') & (df.time != '18:00:00')]
    data = data.drop(data.index[(data.no == 3) & (data.date == '2018/7/16')])
    data = data.drop(data.index[(data.no == 6) & (data.date == '2018/7/16')])
    data.loc[(data['time'] == '9:00:00'), 'time'] = '09:00:00'
    data.loc[(data['time'] == '9:30:00'), 'time'] = '09:30:00'

    # 标签数据
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0
    label = data[y_feature].values.astype(int)
    label = pd.DataFrame(label)

    # 人员特征
    body = data[body_feature].reset_index(drop=True)
    gender = data['gender'].reset_index(drop=True)

    # 环境特征
    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})
    env = data[env_feature].reset_index(drop=True)
    env = pd.concat([env, va], axis=1)

    # 时间
    date = []
    for d in np.array(data[['date', 'time']]):
        date.append(datetime.strptime(d[0] + ' ' + d[1], '%Y/%m/%d %H:%M:%S'))
    date = pd.DataFrame({'date': date})
    return body, gender, env, date, label


def deal_2019_summer():
    df = pd.read_csv('../../dataset/2019_summer_clean.csv').dropna(axis=0, how='any', inplace=False)
    data = df.drop(df.index[(df.time == '12:30:00') | (df.time == '13:00:00') |
                            (df.time == '13:30:00') | (df.time == '14:00:00') |
                            (df.time == '18:00:00')])
    data = data.drop(data.index[(data.no == 9) & (data.date == '2019/7/29')])
    data.loc[(data['time'] == '9:00:00'), 'time'] = '09:00:00'
    data.loc[(data['time'] == '9:30:00'), 'time'] = '09:30:00'
    data = data.sort_values(by=['no', 'date', 'time'], axis=0, ascending=True, inplace=False)

    # 标签数据
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0
    label = data[y_feature].values.astype(int)
    label = pd.DataFrame(label)

    # 人员特征
    body = data[body_feature].reset_index(drop=True)
    gender = data['gender'].reset_index(drop=True)

    # 环境特征
    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})
    env = data[env_feature].reset_index(drop=True)
    env = pd.concat([env, va], axis=1)

    # 时间
    date = []
    for d in np.array(data[['date', 'time']]):
        date.append(datetime.strptime(d[0] + ' ' + d[1], '%Y/%m/%d %H:%M:%S'))
    date = pd.DataFrame({'date': date})
    return body, gender, env, date, label


def deal_2019_winter():
    df = pd.read_csv('../../dataset/2019_winter.csv').dropna(axis=0, how='any', inplace=False)
    data = df.drop(df.index[(df.time == '8:40:00') | (df.time == '8:50:00') | (df.time == '14:10:00')])
    data.loc[(data['time'] == '9:00:00'), 'time'] = '09:00:00'
    data.loc[(data['time'] == '9:30:00'), 'time'] = '09:30:00'
    data.loc[(data['date'] == '2019/1/9'), 'date'] = '2019/1/09'
    data = data.sort_values(by=['no', 'date', 'time'], axis=0, ascending=True, inplace=False)

    # 标签数据
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0
    label = data[y_feature].values.astype(int)
    label = pd.DataFrame(label)

    # 人员特征
    body = data[body_feature].reset_index(drop=True)
    gender = data['gender'].reset_index(drop=True)

    # 环境特征
    va = data['air_speed'].reset_index(drop=True)
    va = pd.DataFrame({'va': va})
    env = data[env_feature].reset_index(drop=True)
    env = pd.concat([env, va], axis=1)

    # 时间
    date = []
    for d in np.array(data[['date', 'time']]):
        date.append(datetime.strptime(d[0] + ' ' + d[1], '%Y/%m/%d %H:%M:%S'))
    date = pd.DataFrame({'date': date})
    return body, gender, env, date, label


def deal_2021_summer():
    df = pd.read_csv('../../dataset/2021.csv').dropna(axis=0, how='any', inplace=False)
    df.loc[(df['time'] == '9:00:00'), 'time'] = '09:00:00'
    df.loc[(df['time'] == '9:30:00'), 'time'] = '09:30:00'
    data = df.drop(df.index[(df.time > '17:30:00') | (df.time >= '12:30:00') & (df.time <= '14:00:00')])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 20) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 56) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 25) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/29') & (data.no == 33) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/25') & (data.no == 49) & (data.time == '12:00:00')])
    data = data.loc[(data['season'] == 'summer')].reset_index(drop=False)

    # 标签数据
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0
    label = data[y_feature].values.astype(int)
    label = pd.DataFrame(label)

    # 人员特征
    body = data[body_feature].reset_index(drop=True)
    gender = data[gender_feature].reset_index(drop=True)

    # 环境特征
    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})
    env = data[env_feature].reset_index(drop=True)
    env = pd.concat([env, va], axis=1)

    # 时间
    date = []
    for d in np.array(data[['date', 'time']]):
        date.append(datetime.strptime(d[0] + ' ' + d[1], '%Y/%m/%d %H:%M:%S'))
    date = pd.DataFrame({'date': date})
    return body, gender, env, date, label


def deal_2021_winter():
    df = pd.read_csv('../../dataset/2021.csv').dropna(axis=0, how='any', inplace=False)
    df.loc[(df['time'] == '9:00:00'), 'time'] = '09:00:00'
    df.loc[(df['time'] == '9:30:00'), 'time'] = '09:30:00'
    data = df.loc[(df['season'] == 'winter')].reset_index(drop=False)

    # 标签数据
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0
    label = data[y_feature].values.astype(int)
    label = pd.DataFrame(label)

    # 人员特征
    body = data[body_feature].reset_index(drop=True)
    gender = data[gender_feature].reset_index(drop=True)

    # 环境特征
    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})
    env = data[env_feature].reset_index(drop=True)
    env = pd.concat([env, va], axis=1)

    # 时间
    date = []
    for d in np.array(data[['date', 'time']]):
        date.append(datetime.strptime(d[0] + ' ' + d[1], '%Y/%m/%d %H:%M:%S'))
    date = pd.DataFrame({'date': date})
    return body, gender, env, date, label

if __name__ == '__main__':

    # 性别特征不需要归一化最后添加
    body_feature = ['age', 'height', 'weight', 'bmi']
    # 风速部分数据集需要自己生成
    env_feature = ['ta', 'hr']
    gender_feature = ['griffith', 'gender', 'sensitivity', 'preference', 'environment']
    y_feature = 'thermal sensation'

    # '''2018 summer'''
    # body1, gender1, env1, date1, label1 = deal_2018()
    #
    # ''' 2019 summer'''
    # body2, gender2, env2,  date2, label2 = deal_2019_summer()

    ''' 2021 summer'''
    body3, gender3, env3, date3, label3 = deal_2021_summer()

    '''save data'''
    # body = pd.concat([body1, body2, body3], axis=0).reset_index(drop=True)
    # gender = pd.concat([gender1, gender2, gender3], axis=0).reset_index(drop=True)
    # env = pd.concat([env1, env2, env3], axis=0).reset_index(drop=True)
    # date = pd.concat([date1, date2, date3], axis=0).reset_index(drop=True)
    # label = pd.concat([label1, label2, label3], axis=0).reset_index(drop=True)
    # season = []
    # for i in range(0, label.shape[0]):
    #     season.append(0)
    # season = pd.DataFrame({'season': season})
    #
    # np.save('./summer/body.npy', body)
    # np.save('./summer/gender.npy', gender)
    # np.save('./summer/env.npy', env)
    # np.save('./summer/date.npy', date)
    # np.save('./summer/season.npy', season)
    # np.save('./summer/label.npy', label)

    # ''' 2019 winter'''
    # body4, gender4, env4, date4, label4 = deal_2019_winter()

    ''' 2021 winter'''
    body5, gender5, env5, date5, label5 = deal_2021_winter()

    '''save data'''
    # body = pd.concat([body4, body5], axis=0).reset_index(drop=True)
    # gender = pd.concat([gender4, gender5], axis=0).reset_index(drop=True)
    # env = pd.concat([env4, env5], axis=0).reset_index(drop=True)
    # date = pd.concat([date4, date5], axis=0).reset_index(drop=True)
    # label = pd.concat([label4, label5], axis=0).reset_index(drop=True)
    #
    # season = []
    # for i in range(0, label.shape[0]):
    #     season.append(1)
    # season = pd.DataFrame({'season': season})
    #
    # np.save('./winter/body.npy', body)
    # np.save('./winter/gender.npy', gender)
    # np.save('./winter/env.npy', env)
    # np.save('./winter/date.npy', date)
    # np.save('./winter/season.npy', season)
    # np.save('./winter/label.npy', label)

    '''save data'''

    # body = pd.concat([body1, body2, body3, body4, body5], axis=0).reset_index(drop=True)
    # gender = pd.concat([gender1, gender2, gender3, gender4, gender5], axis=0).reset_index(drop=True)
    # env = pd.concat([env1, env2, env3, env4, env5], axis=0).reset_index(drop=True)
    # date = pd.concat([date1, date2, date3, date4, date5], axis=0).reset_index(drop=True)
    # label = pd.concat([label1, label2, label3, label4, label5], axis=0).reset_index(drop=True)
    # season = []
    # for i in range(0, label.shape[0]):
    #     season.append(0)
    # season = pd.DataFrame({'season': season})
    #
    # np.save('./all/body.npy', body)
    # np.save('./all/gender.npy', gender)
    # np.save('./all/env.npy', env)
    # np.save('./all/date.npy', date)
    # np.save('./all/season.npy', season)
    # np.save('./all/label.npy', label)
    #

    '''save data'''

    body = pd.concat([body3, body5], axis=0).reset_index(drop=True)
    gender = pd.concat([gender3, gender5], axis=0).reset_index(drop=True)
    env = pd.concat([env3, env5], axis=0).reset_index(drop=True)
    date = pd.concat([date3, date5], axis=0).reset_index(drop=True)
    label = pd.concat([label3, label5], axis=0).reset_index(drop=True)
    season = []
    for i in range(0, label3.shape[0]):
        season.append(0)
    for i in range(label3.shape[0], label.shape[0]):
        season.append(1)
    season = pd.DataFrame({'season': season})

    np.save('./2021/body.npy', body)
    np.save('./2021/gender.npy', gender)
    np.save('./2021/env.npy', env)
    np.save('./2021/date.npy', date)
    np.save('./2021/season.npy', season)
    np.save('./2021/label.npy', label)

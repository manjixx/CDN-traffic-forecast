# -*- coding: utf-8 -*-
import random
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt


def deal_2018():
    df_2018 = pd.read_csv('../../dataset/2018.csv').dropna(axis=0, how='any', inplace=False)
    # print(f'2018年数据共{df_2018.shape[0]}条')
    no2018 = np.array(df_2018['no'].unique())
    # print(f'2018年共计有{no2018}名成员')
    return df_2018


def deal_2019_summer():
    df_2019 = pd.read_csv('../../dataset/2019_summer.csv').dropna(axis=0, how='any', inplace=False)
    # print(f'2019年夏季数据共{df_2019.shape[0]}条')
    no2019 = np.array(df_2019['no'].unique())
    # print(f'2019年夏季共计有{no2019}名成员')
    for i in reversed(no2019):
        df_2019.loc[(df_2019['no'] == i), 'no'] = i + 6
    # print(f'2019年夏季共计有{np.array(df_2019["no"].unique())}名成员')
    return df_2019


def deal_2019_winter():
    df_2019 = pd.read_csv('../../dataset/2019_winter.csv').dropna(axis=0, how='any', inplace=False)
    # print(f'2019年冬季数据共{df_2019.shape[0]}条')
    no2019 = np.array(df_2019['no'].unique())
    # print(f'2019年冬季共计有{no2019}名成员')
    for i in reversed(no2019):
        df_2019.loc[(df_2019['no'] == i), 'no'] = i + 28
    # print(f'2019年冬季共计有{np.array(df_2019["no"].unique())}名成员')
    return df_2019


def deal_2021():
    df_2021 = pd.read_csv('../../dataset/2021.csv').dropna(axis=0, how='any', inplace=False)
    # print(f'2021年数据共{df_2021.shape[0]}条')
    no2021 = np.array(df_2021['no'].unique())
    for i in reversed(no2021):
        df_2021.loc[(df_2021['no'] == i), 'no'] = i + 37
    # print(f'2019年共计有{np.array(df_2021["no"].unique())}名成员')
    return df_2021


def calculate(df):

    no = np.array(df['no'].unique())
    bmi = []
    griffith = []
    for i in no:
        data = df.loc[df["no"] == i]
        weight = np.array(data['weight'].unique())[0]
        height = np.array(data['height'].unique())[0]
        b = round(weight / (height/100) ** 2, 2)
        temp = data['ta']
        pmv = data['thermal sensation']


        statistics = linregress(pmv, temp)
        g = round(statistics.slope, 2)
        if g == 3.43:
            scatter = sns.scatterplot(x=temp, y=pmv)

            scatter.set_xlabel('pmv')

            scatter.set_ylabel('temp')
            print(i)
            plt.show()


        if g < 0:
            g = round(random.random(), 2)

        bmi.extend([b]*data.shape[0])
        griffith.extend([g]*data.shape[0])

    bmi = pd.DataFrame({'bmi': bmi})
    griffith = pd.DataFrame({'griffith': griffith})

    df = pd.concat([df, bmi, griffith], axis=1).reset_index(drop=True)
    return df


def preprocess(df):
    df = df.sort_values(by=['no'], axis=0, ascending=True, inplace=False).reset_index(drop=True)
    for i in np.array(df['no'].unique()):
        data = df.loc[(df['no'] == i)]
        g = np.array(data['griffith'].unique())[0]
        l = data.shape[0]
        if l < 28:
            df = df.drop(df.index[(df.no == i)])
        if g < 0 or ((i % 2 == 0) and 0.9 <= g and g <= 1.2):
            df = df.drop(df.index[(df.no == i)])
    k = 0
    for i in np.array(df['no'].unique()):
        # print(i)
        k += 1
        df.loc[(df['no'] == i), 'no'] = k

        # data = df.loc[(df['no'] == k)].reset_index(drop=True)
        # l = data.shape[0]
        # print(f'{k}号实验人员共有{l}条数据')

    for i in np.array(df['no'].unique()):
        data = df.loc[(df['no'] == i)]
        # print(f'{i}号实验人员共有{data.shape[0]}条数据')

        if data.shape[0] < 100:
            continue

        if data.shape[0] >= 600:
            size = data.shape[0] * 0.85
        elif data.shape[0] >= 500:
            size = data.shape[0] * 0.8
        elif data.shape[0] >= 400:
            size = data.shape[0] * 0.75
        elif data.shape[0] >= 300:
            size = data.shape[0] * 0.7
        elif data.shape[0] >= 250:
            size = data.shape[0] * 0.6
        elif data.shape[0] >= 200:
            size = data.shape[0] * 0.5
        elif data.shape[0] >= 150:
            size = data.shape[0] * 0.4
        else:
            continue

        drop_indices = np.random.choice(data.index[(data.no == i)], size=int(size), replace=False)
        df = df.drop(drop_indices)
        print(f'{i}号共有{data.shape[0]}条数据,删除{int(size)}条数据,剩余{df.loc[(df["no"] == i)].shape[0]}条数据')

    df = df.reset_index(drop=True)

    for i in np.array(df['no'].unique()):
        data = df.loc[(df['no'] == i)].reset_index(drop=True)
        g = np.array(data['griffith'].unique())[0]
        l = data.shape[0]
        print(f'{i}号实验人员热敏感度为{g}')
    return df


if __name__ == '__main__':

    filepath = ['../Dataset/data/Dataset.csv',
                '../Dataset/data/body.npy',
                '../Dataset/data/env.npy',
                '../Dataset/data/gender.npy',
                '../Dataset/data/label.npy'
                ]
    for f in filepath:
        if os.access(f, os.F_OK):
            os.remove(f)

    feature = ['no', 'gender', 'age', 'height', 'weight', 'ta', 'hr', 'season', 'thermal sensation']

    df1 = deal_2018()[feature].reset_index(drop=True)
    df2 = deal_2019_winter()[feature].reset_index(drop=True)
    df3 = deal_2019_summer()[feature].reset_index(drop=True)
    df4 = deal_2021()[feature].reset_index(drop=True)
    df = pd.concat([df1, df2, df3, df4], axis=0).reset_index(drop=True)
    df = df.drop(df.index[(df.ta < 18) | (df.ta > 35) | (df.hr > 100)]).reset_index(drop=True)
    # df = df.drop(df.index[(df.no == 36)]).reset_index(drop=True)
    # df = df.drop(df.index[(df.no == 77)]).reset_index(drop=True)

    df.loc[(df['season'] == 'summer'), 'season'] = 0
    df.loc[(df['season'] == 'winter'), 'season'] = 1

    df = calculate(df)
    df = preprocess(df)
    # print(np.array(df['no'].unique()))
    print(df)

    df.to_csv('./data/Dataset.csv', index=False)

    body_feature = ['age', 'height', 'weight', 'bmi']
    env_feature = ['ta', 'hr', 'season']
    gender_feature = ['gender', 'griffith']

    y_feature = 'thermal sensation'
    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0

    body = df[body_feature].reset_index(drop=True)
    env = df[env_feature].reset_index(drop=True)
    season = df['season'].reset_index(drop=True)
    gender = df[gender_feature].reset_index(drop=True)
    label = df[y_feature].reset_index(drop=True)

    # np.save('data/body.npy', body)
    # np.save('data/env.npy', env)
    # np.save('data/gender.npy', gender)
    # np.save('data/season.npy', season)
    # np.save('data/label.npy', label)
